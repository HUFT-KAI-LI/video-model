"""R0: prompt-conditioned soft context fusion; no video-state retrieval or latent replacement."""
import torch
from torch import nn
from torch.nn import functional as F
from .reality_encoder import MemoryProjector


def reference_dropout(mask, generator, probability, training=True):
    if not 0 <= probability <= 1:
        raise ValueError("Dropout probability outside [0,1]")
    if not training or probability == 0:
        return mask.clone()
    return mask & (torch.rand(mask.shape, generator=generator, device=mask.device) >= probability)


class RealityMemory(nn.Module):
    def __init__(self, feature_dim=384, memory_dim=256, context_dim=4096, heads=4, gate_init_logit=-2.):
        super().__init__()
        if memory_dim % heads:
            raise ValueError("Memory dimension must be divisible by heads")
        self.projector = MemoryProjector(feature_dim, memory_dim)
        self.query = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, memory_dim))
        self.key, self.value = nn.Linear(memory_dim, memory_dim), nn.Linear(memory_dim, memory_dim)
        self.output = nn.Linear(memory_dim, context_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.gate = nn.Sequential(nn.Linear(2 * memory_dim + 1, 128), nn.SiLU(), nn.Linear(128, 1))
        nn.init.constant_(self.gate[-1].bias, gate_init_logit)
        self.heads = heads

    def forward(self, context, features, reference_mask):
        """No timestamp/kind input. Empty or fully dropped memory is structurally inert.

        Returns fused B,L,C context and differentiable scalar diagnostics.
        All parameters stay in the graph even on a no-memory DDP rank.
        """
        if features.ndim != 4 or reference_mask.shape != features.shape[:2] or reference_mask.dtype != torch.bool:
            raise ValueError("Expected B,K,P,D features and a boolean B,K mask")
        if context.shape[0] != features.shape[0]:
            raise ValueError("Context/reference batch sizes differ")
        if features.shape[1] == 0:
            features = features.new_zeros(features.shape[0], 1, features.shape[2], features.shape[3])
            reference_mask = reference_mask.new_zeros(features.shape[:2])
        # Invalid padding must not leak into key/value, even if it contains NaNs.
        features = torch.where(reference_mask[:, :, None, None], features, torch.zeros_like(features))
        memory = self.projector(features).flatten(1, 2)
        token_mask = reference_mask.repeat_interleave(features.shape[2], dim=1)
        prompt_mask = context.detach().abs().sum(-1) > 0  # UMT5 wrapper zeros padding.
        q = self.query(context)
        k, v = self.key(memory), self.value(memory)
        def heads(x):
            return x.unflatten(-1, (self.heads, x.shape[-1] // self.heads)).transpose(1, 2)
        scores = (heads(q).float() @ heads(k).float().transpose(-2, -1)) / (q.shape[-1] // self.heads) ** .5
        weights = scores.masked_fill(~token_mask[:, None, None], -1e4).softmax(-1)
        weights = weights * token_mask[:, None, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        retrieved = (weights.to(v.dtype) @ heads(v)).transpose(1, 2).flatten(2)
        denom = prompt_mask.sum(1, keepdim=True).clamp_min(1)
        pooled_q = (q * prompt_mask[..., None]).sum(1) / denom
        pooled_r = (retrieved * prompt_mask[..., None]).sum(1) / denom
        similarity = F.cosine_similarity(pooled_q.float(), pooled_r.float(), dim=-1)[:, None].to(q.dtype)
        gate = self.gate(torch.cat((pooled_q, pooled_r, similarity), -1)).sigmoid().flatten()
        active = reference_mask.any(1)
        gate = gate * active
        raw_delta = self.output(retrieved) * prompt_mask[..., None] * active[:, None, None]
        delta = gate[:, None, None] * raw_delta
        fused = torch.where(active[:, None, None], context + delta.to(context.dtype), context)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1).mean(1)
        entropy = (entropy * prompt_mask).sum(1) / denom.flatten()
        valid_tokens = token_mask.sum(1)
        # N=0/1 has no retrieval uncertainty; define normalized entropy as zero.
        normalized_entropy = torch.where(valid_tokens > 1,
                                         entropy / valid_tokens.float().clamp_min(2).log(),
                                         torch.zeros_like(entropy))
        return fused, {"gate": gate, "active": active,
                       "relevance_score": similarity.float().flatten() * active,
                       "raw_delta_norm": raw_delta.float().square().mean(dim=(-1, -2)).sqrt(),
                       "applied_delta_norm": delta.float().square().mean(dim=(-1, -2)).sqrt(),
                       "delta_square": raw_delta.float().square().mean(),
                       "attention_entropy": entropy, "attention_entropy_normalized": normalized_entropy,
                       "valid_memory_tokens": valid_tokens}


def memory_regularization(stats, wrong_reference, delta_weight, wrong_weight):
    eligible = wrong_reference.bool() & stats["active"]
    wrong = (stats["gate"].float().square() * eligible).sum() / eligible.sum().clamp_min(1)
    return delta_weight * stats["delta_square"] + wrong_weight * wrong, wrong
