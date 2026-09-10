"""Frozen R1-A Top-1 routing, preserving every token of one reference.

Cosine scores select a reference; softmax weights are diagnostics only. Raw
feature scaling would be erased by the projector's LayerNorm, and patch-wise
averaging across views has no spatial correspondence guarantee.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenStateRouter(nn.Module):
    def __init__(self, temperature=0.1, mode="top1"):
        super().__init__()
        if mode != "top1":
            raise ValueError("R1-A supports only top1; soft/topk require reference-preserving attention priors")
        temperature = float(temperature)
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
            raise ValueError("Router temperature must be positive and finite")
        self.temperature, self.mode = temperature, mode

    @staticmethod
    def pooled(features):
        return features.float().mean(-2)

    @torch.no_grad()
    def forward(self, prefix_features, candidate_features, candidate_mask=None):
        """Return (B,1,P,D) memory, boolean (B,1) mask and detached diagnostics.

        Prefix: (Pq,D) or (B,Pq,D). Candidates: (K,P,D) or (B,K,P,D).
        All-masked rows return zero memory and selected index -1. Invalid padding
        may contain NaNs; valid features and prefix features must be finite.
        """
        prefix = prefix_features[None] if prefix_features.ndim == 2 else prefix_features
        candidates = candidate_features[None] if candidate_features.ndim == 3 else candidate_features
        if prefix.ndim != 3 or candidates.ndim != 4:
            raise ValueError("Expected (Pq,D)/(B,Pq,D) prefix and (K,P,D)/(B,K,P,D) candidates")
        if prefix.shape[0] != candidates.shape[0] or prefix.shape[-1] != candidates.shape[-1]:
            raise ValueError("Prefix and candidate batch/feature dimensions differ")
        if not all(prefix.shape) or not all(candidates.shape):
            raise ValueError("Router requires nonempty batches, references and tokens")
        if prefix.device != candidates.device:
            raise ValueError("Prefix and candidates must be on the same device")
        if candidate_mask is None:
            mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        else:
            mask = candidate_mask.to(device=candidates.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask[None].expand(candidates.shape[0], -1)
            if mask.shape != candidates.shape[:2]:
                raise ValueError("Candidate mask shape does not match candidates")
        clean = torch.where(mask[..., None, None], candidates, torch.zeros_like(candidates))
        if not torch.isfinite(prefix).all() or not torch.isfinite(clean).all():
            raise ValueError("Valid router features must be finite")
        scores = F.cosine_similarity(self.pooled(prefix)[:, None], self.pooled(clean), dim=-1)
        active = mask.any(-1)
        # Argmax uses scores directly: temperature cannot change the selected ID.
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        selection = masked_scores.argmax(-1, keepdim=True)
        logits = torch.where(active[:, None], masked_scores, torch.zeros_like(scores))
        logits = (logits - logits.max(-1, keepdim=True).values) / self.temperature
        weights = logits.softmax(-1) * mask
        memory = torch.gather(clean, 1, selection[..., None, None].expand(-1, -1, *clean.shape[2:]))
        memory = torch.where(active[:, None, None, None], memory, torch.zeros_like(memory))
        return memory, active[:, None], {
            "router_scores": scores.masked_fill(~mask, -1e4), "router_weights": weights,
            "active": active, "selected_indices": selection.masked_fill(~active[:, None], -1),
            "selected_weights": active[:, None].to(weights.dtype),
        }


def route_from_encoded(router, prefix_features, candidate_features, candidate_mask=None):
    return router(prefix_features, candidate_features, candidate_mask)
