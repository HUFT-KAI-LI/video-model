"""R1-A frozen state router: the visible-prefix DINO state directly determines
which reference tokens reach the generation residual.

Pipeline (the router itself is never trained):

    visible prefix pixels -> frozen DINO -> q_t = Pool(tokens)
    candidate references  -> cached DINO tokens -> m_k = Pool(tokens)
    s_k = cos(q_t, m_k);  w = softmax(s / tau)        (mask-aware)
    soft mode:  M~ = sum_k w_k M_k                     -> one effective memory slot
    topk mode:  keep the k highest-weight candidates, renormalized

The routed memory is exactly what ``RealityMemory`` consumes, so retrieval is
structurally coupled to generation instead of being an auxiliary score: the
weights decide which reference tokens enter the context residual. R1-A trains the
memory-to-context adapter while this router (and the frozen DINO encoder feeding
it) stays fixed.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FrozenStateRouter(nn.Module):
    def __init__(self, temperature=0.1, mode="soft", top_k=2):
        super().__init__()
        if mode not in ("soft", "topk"):
            raise ValueError("Router mode must be 'soft' or 'topk'")
        temperature = float(temperature)
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
            raise ValueError("Router temperature must be positive and finite")
        if mode == "topk" and (type(top_k) is not int or top_k < 1):
            raise ValueError("topk mode requires a positive integer top_k")
        self.temperature, self.mode, self.top_k = temperature, mode, top_k

    @staticmethod
    def pooled(features):
        """(..., P, D) -> (..., D) by averaging memory tokens."""
        return features.float().mean(-2)

    @torch.no_grad()
    def forward(self, prefix_features, candidate_features, candidate_mask=None):
        """Return ``(memory, mask, stats)`` ready for ``RealityMemory.forward``.

        ``prefix_features``: (Pq, D) or (B, Pq, D) pooled-prefix state.
        ``candidate_features``: (K, P, D) or (B, K, P, D) reference tokens.
        ``candidate_mask``: optional boolean (K,) or (B, K); invalid candidates
        receive zero weight and never enter the residual.
        """
        prefix = prefix_features[None] if prefix_features.ndim == 2 else prefix_features
        candidates = candidate_features[None] if candidate_features.ndim == 3 else candidate_features
        if prefix.ndim != 3 or candidates.ndim != 4:
            raise ValueError("Expected (Pq,D)/(B,Pq,D) prefix and (K,P,D)/(B,K,P,D) candidates")
        if prefix.shape[0] != candidates.shape[0]:
            raise ValueError("Prefix and candidate batch sizes differ")
        if prefix.shape[-1] != candidates.shape[-1]:
            raise ValueError("Prefix and candidate feature dimensions differ")
        count = candidates.shape[1]
        if count == 0:
            raise ValueError("Router requires at least one candidate reference")
        if candidate_mask is None:
            mask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
        else:
            mask = candidate_mask.to(torch.bool)
            if mask.ndim == 1:
                mask = mask[None].expand(candidates.shape[0], -1)
            if mask.shape != candidates.shape[:2]:
                raise ValueError("Candidate mask shape does not match candidates")
        scores = F.cosine_similarity(self.pooled(prefix)[:, None, :], self.pooled(candidates), dim=-1)
        scores = scores.masked_fill(~mask, -1e4)
        weights = torch.softmax(scores / self.temperature, dim=-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        active = mask.any(-1)
        if self.mode == "soft":
            memory = (weights[..., None, None] * candidates.float()).sum(1, keepdim=True)
            output_mask = active[:, None]
            stats = {"router_scores": scores, "router_weights": weights, "active": active}
        else:
            valid = mask.sum(-1)
            if int(valid.min()) < 1:
                raise ValueError("topk routing requires at least one valid candidate per batch row")
            top_k = min(self.top_k, int(valid.min()))
            selection = torch.topk(weights, top_k, dim=-1).indices
            gathered = torch.gather(candidates, 1, selection[..., None, None].expand(-1, -1, candidates.shape[2], candidates.shape[3]))
            selected = torch.gather(weights, 1, selection)
            selected = selected / selected.sum(-1, keepdim=True).clamp_min(1e-8)
            memory = gathered.float() * selected[..., None, None]
            output_mask = torch.gather(mask, 1, selection)
            stats = {"router_scores": scores, "router_weights": weights, "active": active,
                     "selected_indices": selection, "selected_weights": selected}
        return memory.to(candidate_features.dtype), output_mask, stats


def route_from_encoded(router, prefix_features, candidate_features, candidate_mask=None):
    """Convenience wrapper: route pooled prefix state onto reference tokens."""
    return router(prefix_features, candidate_features, candidate_mask)
