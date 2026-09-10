"""Deterministic, dependency-light metrics for the Edit-Ready MVP.

No new heavy dependencies: CLIP / VLM text-alignment is *not* available offline in
this environment (HuggingFace is unreachable), so edit responsiveness is reported
with explicit, interpretable proxies plus a frozen DINOv2 feature distance for
appearance change and boundary continuity.  ``EDIT_READY_MVP.md`` states this
limitation in the results section instead of pretending a text metric exists.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from .edit_replay import chunk_frame_slice


# --------------------------------------------------------------------------- #
# tensor distances
# --------------------------------------------------------------------------- #
def latent_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    x, y = a.float().flatten(), b.float().flatten()
    difference = x - y
    denominator = x.norm() * y.norm()
    return {"mse": float(difference.square().mean().item()),
            "rmse": float(difference.square().mean().sqrt().item()),
            "max_abs": float(difference.abs().max().item()),
            "cosine": float((x @ y / denominator).item()) if denominator > 0 else 1.0,
            "relative_l2": float((difference.norm() / x.norm()).item()) if x.norm() > 0 else 0.0,
            "exact": bool(torch.equal(a, b))}


def pixel_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    x, y = a.float(), b.float()
    mse = float((x - y).square().mean().item())
    return {"mse": mse, "psnr": float("inf") if mse == 0 else float(10 * math.log10(1.0 / mse)),
            "max_abs": float((x - y).abs().max().item()),
            "exact": bool(torch.equal(a, b))}


def frame_change(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Mean absolute pixel change between two frame tensors in [0, 1]."""
    x, y = a.float(), b.float()
    return {"mean_abs": float((x - y).abs().mean().item()),
            "rmse": float((x - y).square().mean().sqrt().item())}


# --------------------------------------------------------------------------- #
# boundary continuity
# --------------------------------------------------------------------------- #
def boundary_latent(latents: torch.Tensor, chunk_index: int, num_frame_per_block: int) -> Dict[str, float]:
    """D_L / D_R on the latent boundary around ``chunk_index``.

    The base (uninterrupted) boundary distance is computed from the same tensor
    when the caller passes the original video, so deltas stay comparable.
    """
    block = int(num_frame_per_block)
    start = chunk_index * block
    left = latents[:, start - 1:start].float() if start > 0 else None
    first = latents[:, start:start + 1].float()
    last = latents[:, start + block - 1:start + block].float()
    right = latents[:, start + block:start + block + 1].float()
    values: Dict[str, float] = {}
    values["left_mse"] = float((left - first).square().mean().item()) if left is not None else float("nan")
    values["right_mse"] = float((right - last).square().mean().item()) if right.numel() else float("nan")
    return values


def boundary_pixels(pixels: torch.Tensor, chunk_index: int, num_frame_per_block: int) -> Dict[str, float]:
    """D_L / D_R on the decoded pixel boundary around ``chunk_index``."""
    total = pixels.shape[1]
    span = chunk_frame_slice(chunk_index, num_frame_per_block, total)
    start, end = span["pixel_start"], span["pixel_end"]
    values: Dict[str, float] = {}
    left = pixels[:, start - 1:start].float() if start > 0 else None
    first = pixels[:, start:start + 1].float()
    last = pixels[:, end - 1:end].float()
    right = pixels[:, end:end + 1].float()
    values["left_mse"] = float((left - first).square().mean().item()) if left is not None else float("nan")
    values["right_mse"] = float((right - last).square().mean().item()) if right.numel() else float("nan")
    return values


def boundary_delta(edited: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    return {f"delta_{key}": edited[key] - base.get(key, float("nan")) for key in edited}


# --------------------------------------------------------------------------- #
# colour / luma proxies
# --------------------------------------------------------------------------- #
_COLOR_RANGES: Dict[str, Sequence[Sequence[float]]] = {
    # name -> list of (hue_min, hue_max) in degrees (wrapping allowed)
    "red": [(345, 360), (0, 20)],
    "orange": [(20, 40)],
    "yellow": [(40, 70)],
    "green": [(70, 165)],
    "cyan": [(165, 195)],
    "blue": [(195, 260)],
    "purple": [(260, 300)],
    "magenta": [(300, 345)],
}


def to_uint8(frames: torch.Tensor) -> np.ndarray:
    """[T, C, H, W] or [T, H, W, C] float in [0, 1] -> uint8 HWC array."""
    array = frames.detach().float().cpu()
    if array.ndim == 4 and array.shape[1] == 3:
        array = array.permute(0, 2, 3, 1)
    if array.ndim != 4:
        raise ValueError(f"Expected 4D frames, got {tuple(array.shape)}")
    return (array.clamp(0, 1) * 255.0).round().byte().numpy()


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    maximum = values.max(axis=-1)
    minimum = values.min(axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red_max = nonzero & (maximum == red)
    green_max = nonzero & (maximum == green) & ~red_max
    blue_max = nonzero & (maximum == blue) & ~red_max & ~green_max
    hue[red_max] = (60 * ((green - blue) / np.where(delta == 0, 1, delta)))[red_max] % 360
    hue[green_max] = (60 * ((blue - red) / np.where(delta == 0, 1, delta)) + 120)[green_max]
    hue[blue_max] = (60 * ((red - green) / np.where(delta == 0, 1, delta)) + 240)[blue_max]
    saturation = np.where(maximum > 1e-6, delta / np.where(maximum == 0, 1, maximum), 0.0)
    return np.stack([hue, saturation, maximum], axis=-1)


def color_fraction(frames: torch.Tensor, color: str, *, min_saturation: float = 0.35,
                   min_value: float = 0.15, max_value: float = 1.01) -> float:
    """Fraction of pixels whose hue falls in ``color``'s range."""
    if color == "black":
        hsv = rgb_to_hsv(to_uint8(frames))
        return float((hsv[..., 2] < min_value).mean())
    if color == "white":
        hsv = rgb_to_hsv(to_uint8(frames))
        return float(((hsv[..., 2] > 0.85) & (hsv[..., 1] < 0.15)).mean())
    if color not in _COLOR_RANGES:
        raise ValueError(f"Unknown colour probe {color!r}")
    hsv = rgb_to_hsv(to_uint8(frames))
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = np.zeros_like(hue, dtype=bool)
    for low, high in _COLOR_RANGES[color]:
        mask |= (hue >= low) & (hue < high + 1e-6)
    mask &= saturation >= min_saturation
    mask &= (value >= min_value) & (value <= max_value)
    return float(mask.mean())


def mean_luma(frames: torch.Tensor) -> float:
    array = to_uint8(frames).astype(np.float32)
    return float((0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]).mean() / 255.0)


def mean_saturation(frames: torch.Tensor) -> float:
    return float(rgb_to_hsv(to_uint8(frames))[..., 1].mean())


def color_histogram(frames: torch.Tensor, bins: int = 12) -> np.ndarray:
    hsv = rgb_to_hsv(to_uint8(frames))
    mask = (hsv[..., 1] > 0.25) & (hsv[..., 2] > 0.15)
    histogram, _ = np.histogram(hsv[..., 0][mask], bins=bins, range=(0, 360))
    total = histogram.sum()
    return (histogram / total) if total else histogram.astype(np.float64)


def histogram_distance(a: torch.Tensor, b: torch.Tensor, bins: int = 12) -> float:
    return float(np.abs(color_histogram(a, bins) - color_histogram(b, bins)).sum() / 2)


# --------------------------------------------------------------------------- #
# probe evaluation
# --------------------------------------------------------------------------- #
def evaluate_probe(probe: Optional[Dict[str, Any]], base_frames: torch.Tensor,
                   edited_frames: torch.Tensor) -> Dict[str, Any]:
    """Directional, interpretable response of an edit prompt.

    ``S_proxy`` is signed so that a positive value always means "the local edit
    moved the chunk in the direction the new prompt asked for".  It is *not* a
    text-alignment score.
    """
    if not probe:
        return {"kind": "none", "S_proxy": 0.0, "measurements": {}}
    kind = probe.get("kind", "change")
    measurements: Dict[str, float] = {}
    if kind == "color_fraction":
        target = str(probe["color"])
        reference = probe.get("reference_color")
        target_base = color_fraction(base_frames, target)
        target_edited = color_fraction(edited_frames, target)
        measurements.update({"target_fraction_base": target_base,
                             "target_fraction_edited": target_edited,
                             "delta_target_fraction": target_edited - target_base})
        score = target_edited - target_base
        if reference:
            reference_base = color_fraction(base_frames, str(reference))
            reference_edited = color_fraction(edited_frames, str(reference))
            measurements.update({"reference_fraction_base": reference_base,
                                 "reference_fraction_edited": reference_edited,
                                 "delta_reference_fraction": reference_edited - reference_base})
            score = measurements["delta_target_fraction"] - measurements["delta_reference_fraction"]
        measurements.update({"histogram_distance": histogram_distance(base_frames, edited_frames),
                             "appearance_change": frame_change(base_frames, edited_frames)["mean_abs"]})
        return {"kind": kind, "S_proxy": float(score), "measurements": measurements}
    if kind == "luma":
        base_luma, edited_luma = mean_luma(base_frames), mean_luma(edited_frames)
        delta = edited_luma - base_luma
        direction = str(probe.get("direction", "decrease"))
        score = -delta if direction == "decrease" else delta
        measurements.update({"luma_base": base_luma, "luma_edited": edited_luma, "delta_luma": delta,
                             "appearance_change": frame_change(base_frames, edited_frames)["mean_abs"]})
        return {"kind": kind, "S_proxy": float(score), "measurements": measurements}
    if kind == "saturation":
        base_sat, edited_sat = mean_saturation(base_frames), mean_saturation(edited_frames)
        delta = edited_sat - base_sat
        direction = str(probe.get("direction", "increase"))
        score = delta if direction == "increase" else -delta
        measurements.update({"saturation_base": base_sat, "saturation_edited": edited_sat,
                             "delta_saturation": delta,
                             "appearance_change": frame_change(base_frames, edited_frames)["mean_abs"]})
        return {"kind": kind, "S_proxy": float(score), "measurements": measurements}
    if kind == "change":
        measurements.update({"appearance_change": frame_change(base_frames, edited_frames)["mean_abs"],
                             "histogram_distance": histogram_distance(base_frames, edited_frames)})
        return {"kind": kind, "S_proxy": float(measurements["appearance_change"]),
                "measurements": measurements}
    raise ValueError(f"Unknown probe kind {kind!r}")


# --------------------------------------------------------------------------- #
# optional frozen DINOv2 feature distance
# --------------------------------------------------------------------------- #
class DinoFeatureDistance:
    """Frozen DINOv2-S CLS/patch features for appearance + boundary distance.

    Optional: loads only when ``--dino`` is requested and the local snapshot
    exists, so the CPU test-suite never needs model weights.
    """

    def __init__(self, path, device: str | torch.device = "cpu", image_size: int = 224):
        from transformers import AutoImageProcessor, Dinov2Model

        self.backbone = Dinov2Model.from_pretrained(str(path), local_files_only=True)
        self.backbone.eval().requires_grad_(False).to(device)
        self.processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True,
                                                            use_fast=False)
        self.device = torch.device(device)
        self.image_size = image_size
        self.identity = {"path": str(path), "image_size": image_size,
                         "pool": "mean-patch-tokens"}

    @torch.no_grad()
    def features(self, frames: torch.Tensor) -> torch.Tensor:
        """[T, C, H, W] float in [0, 1] -> [T, D] mean patch features."""
        if frames.dtype != torch.uint8:
            pixels = (frames.detach().float().clamp(0, 1).cpu() * 255.0).round().to(torch.uint8)
        else:
            pixels = frames.detach().cpu()
        inputs = self.processor(images=[frame.permute(1, 2, 0).numpy() for frame in pixels],
                                return_tensors="pt", do_resize=True,
                                size={"height": self.image_size, "width": self.image_size})
        outputs = self.backbone(pixel_values=inputs["pixel_values"].to(self.device))
        return outputs.last_hidden_state[:, 1:].mean(dim=1)

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
        features_a, features_b = self.features(a), self.features(b)
        cosine = torch.nn.functional.cosine_similarity(features_a, features_b, dim=-1)
        return {"mean_cosine": float(cosine.mean().item()),
                "mean_cosine_distance": float((1 - cosine).mean().item())}
