"""Features and small controllers for M1-A oracle-mask distillation."""
from __future__ import annotations

import hashlib
from typing import Any, Dict

import torch
from torch import nn

from . import edit_replay as er


PROTOCOL = "oracle_mask_distillation_m1a_v1"
PROMPT_DIM = 64
STATE_FEATURES_PER_LAYER = 24
LAYERS = 30


@torch.no_grad()
def prompt_delta_feature(pipeline, base_prompt: str, edit_prompt: str) -> torch.Tensor:
    """Fixed projection of mean pooled frozen-T5 edit-minus-base embeddings."""
    base = er.make_conditioning(pipeline, base_prompt)["prompt_embeds"].float().mean(dim=1)[0].cpu()
    edit = er.make_conditioning(pipeline, edit_prompt)["prompt_embeds"].float().mean(dim=1)[0].cpu()
    generator = torch.Generator(device="cpu").manual_seed(1701)
    projection = (torch.randint(0, 2, (base.numel(), PROMPT_DIM), generator=generator,
                                dtype=torch.int8).float() * 2 - 1)
    return (edit - base) @ projection / base.numel() ** 0.5


def checkpoint_state_feature(checkpoint: Any) -> torch.Tensor:
    """Pool active history V into per-layer per-head mean and RMS summaries."""
    if hasattr(checkpoint, "kv_cache"):
        kv_cache = checkpoint.kv_cache
        frame_tokens = int(checkpoint.frame_seq_length)
        sink_frames = int(checkpoint.sink_size)
        local_frames = int(checkpoint.local_attn_size)
        block = int(checkpoint.num_frame_per_block)
    else:
        kv_cache = checkpoint["kv_cache"]
        frame_tokens = int(checkpoint["frame_seq_length"])
        sink_frames = int(checkpoint["sink_size"])
        local_frames = int(checkpoint["local_attn_size"])
        block = int(checkpoint["num_frame_per_block"])
    if len(kv_cache) != LAYERS:
        raise ValueError(f"expected {LAYERS} KV layers, got {len(kv_cache)}")
    local_history_frames = local_frames - sink_frames - block
    if (sink_frames, local_history_frames) != (3, 6):
        raise ValueError("M1-A requires the frozen 3 sink + 6 local history frames")
    sink_tokens, local_tokens = sink_frames * frame_tokens, local_history_frames * frame_tokens
    summaries = []
    for layer in kv_cache:
        values = layer["v"].float().cpu()
        end = int(layer["local_end_index"].flatten()[0])
        active = torch.cat([values[:, :sink_tokens], values[:, end - local_tokens:end]], dim=1)
        if active.shape[1] != 9 * frame_tokens or active.shape[2:] != (12, 128):
            raise ValueError(f"unexpected active history V shape {tuple(active.shape)}")
        mean = active.mean(dim=(0, 1, 3))
        rms = active.square().mean(dim=(0, 1, 3)).sqrt()
        summaries.append(torch.cat([mean, rms]))
    result = torch.stack(summaries)
    if result.shape != (LAYERS, STATE_FEATURES_PER_LAYER) or not torch.isfinite(result).all():
        raise ValueError("invalid pooled V-cache state feature")
    return result


def feature_digest(prompt: torch.Tensor, state: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in (prompt, state):
        digest.update(value.detach().float().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PromptOnlyController(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(PROMPT_DIM, 64), nn.GELU(), nn.Linear(64, LAYERS))

    def forward(self, prompt):
        return torch.sigmoid(self.network(prompt))


class PromptStateController(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_embedding = nn.Embedding(LAYERS, 8)
        self.network = nn.Sequential(
            nn.Linear(PROMPT_DIM + STATE_FEATURES_PER_LAYER + 8, 64), nn.GELU(),
            nn.Linear(64, 1))

    def forward(self, prompt, state):
        batch = prompt.shape[0]
        layer_ids = torch.arange(LAYERS, device=prompt.device)
        layer = self.layer_embedding(layer_ids).unsqueeze(0).expand(batch, -1, -1)
        prompt = prompt.unsqueeze(1).expand(-1, LAYERS, -1)
        return torch.sigmoid(self.network(torch.cat([prompt, state, layer], dim=-1)).squeeze(-1))


def normalize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value - mean) / std.clamp_min(1e-6)


def load_controllers(path, device="cpu") -> Dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("controller checkpoint protocol mismatch")
    prompt_only, prompt_state = PromptOnlyController(), PromptStateController()
    prompt_only.load_state_dict(payload["prompt_only_state_dict"], strict=True)
    prompt_state.load_state_dict(payload["prompt_state_state_dict"], strict=True)
    prompt_only.eval().requires_grad_(False).to(device)
    prompt_state.eval().requires_grad_(False).to(device)
    return {"prompt_only": prompt_only, "prompt_state": prompt_state,
            "normalization": payload["normalization"], "metadata": payload["metadata"]}
