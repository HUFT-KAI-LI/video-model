"""Video/frame IO helpers for the Edit-Ready MVP (PyAV, no ffmpeg binary needed)."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

from .edit_metrics import to_uint8


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """[T, C, H, W] float in [0, 1] -> [T, H, W, C] uint8."""
    return to_uint8(frames)


def label_frames(frames: np.ndarray, label: str, box_width: int = 240, box_height: int = 28) -> np.ndarray:
    """Burn a text header into every frame so side-by-side comparisons read clearly."""
    array = frames.copy()
    for index in range(array.shape[0]):
        picture = Image.fromarray(array[index])
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, box_width, box_height), fill="black")
        draw.text((5, 6), label, fill="white")
        array[index] = np.asarray(picture)
    return array


def write_video(path, frames: np.ndarray, fps: float) -> None:
    """Write a uint8 ``[T, H, W, C]`` array as H.264 mp4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4:
        raise ValueError(f"Expected [T, H, W, C] frames, got {tuple(frames.shape)}")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(float(fps)).limit_denominator(1000))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for pixels in frames:
            picture = Image.fromarray(pixels)
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(picture), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_comparison(path, variants: Dict[str, torch.Tensor], fps: float,
                     labels: Optional[Dict[str, str]] = None) -> None:
    """Concatenate several [1, T, C, H, W] videos side by side with burned-in labels.

    Note: this is a *review artifact*.  All numeric comparisons use tensors, never
    decoded/re-encoded mp4 bytes (plan section 10.3).
    """
    arrays = []
    for name, value in variants.items():
        frames = value[0] if value.ndim == 5 else value
        label = (labels or {}).get(name, name)
        arrays.append(label_frames(frames_to_uint8(frames), label))
    length = min(array.shape[0] for array in arrays)
    combined = np.concatenate([array[:length] for array in arrays], axis=2)
    write_video(path, combined, fps)


def write_chunk_strip(path, variants: Dict[str, torch.Tensor], start: int, end: int, fps: float) -> None:
    """Review artifact: only the edited chunk's pixel frames."""
    trimmed = {name: (value[0] if value.ndim == 5 else value)[start:end]
               for name, value in variants.items()}
    write_comparison(path, trimmed, fps)
