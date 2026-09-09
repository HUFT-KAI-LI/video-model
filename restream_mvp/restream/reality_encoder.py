"""Frozen DINOv2 patch features and a small trainable memory projector."""
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .reality_data import canonical_hash


def encoder_identity(path, image_size, num_tokens):
    path = Path(path)
    provenance = json.loads((path / "provenance.json").read_text())
    for name, expected in provenance["sha256"].items():
        with (path / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Encoder checksum mismatch: {name}")
    return {"schema": 1, "weights": provenance["sha256"], "image_size": image_size,
            "num_tokens": num_tokens, "preprocess": f"AutoImageProcessor/PIL-RGB/transformers-{version('transformers')}",
            "pool": "spatial-adaptive-average-patch-tokens/no-cls"}


class MemoryProjector(nn.Module):
    def __init__(self, feature_dim, memory_dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, memory_dim),
                                 nn.GELU(), nn.Linear(memory_dim, memory_dim))

    def forward(self, features):
        return self.net(features)


class RealityEncoder(nn.Module):
    def __init__(self, path, output_dim=256, num_memory_tokens=8, image_size=224):
        super().__init__()
        from transformers import AutoImageProcessor, Dinov2Model
        self.backbone = Dinov2Model.from_pretrained(str(path), local_files_only=True).eval().requires_grad_(False)
        self.processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True, use_fast=False)
        self.num_memory_tokens, self.image_size = num_memory_tokens, image_size
        if image_size % self.backbone.config.patch_size:
            raise ValueError("DINO image size must be a multiple of patch size")
        self.identity = encoder_identity(path, image_size, num_memory_tokens)
        self.fingerprint = canonical_hash(self.identity)
        self.projector = MemoryProjector(self.backbone.config.hidden_size, output_dim)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def encode_visual(self, images):
        """B,K,3,H,W RGB [0,1] -> B,K,P,D raw features; no timestamps."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("Expected finite B,K,3,H,W RGB images in [0,1]")
        batch, count = images.shape[:2]
        if count == 0:
            return images.new_empty(batch, 0, self.num_memory_tokens, self.backbone.config.hidden_size)
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Expected finite RGB in [0,1]")
        from PIL import Image
        rgb = (images.flatten(0, 1).float().cpu().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
        processed = self.processor(images=[Image.fromarray(x) for x in rgb],
                                   size={"shortest_edge": round(self.image_size * 256 / 224)},
                                   crop_size={"height": self.image_size, "width": self.image_size}, return_tensors="pt")
        pixels = processed["pixel_values"].to(next(self.backbone.parameters()))
        features = self.backbone(pixel_values=pixels).last_hidden_state[:, 1:]
        side = math.isqrt(features.shape[1])
        if side * side != features.shape[1]:
            raise ValueError("Expected a square DINO patch grid")
        rows = math.isqrt(self.num_memory_tokens)
        while self.num_memory_tokens % rows:
            rows -= 1
        grid = features.transpose(1, 2).reshape(batch * count, -1, side, side)
        pooled = F.adaptive_avg_pool2d(grid.float(), (rows, self.num_memory_tokens // rows))
        return pooled.flatten(2).transpose(1, 2).reshape(batch, count, self.num_memory_tokens, -1)

    def project(self, features):
        return self.projector(features).flatten(1, 2)
