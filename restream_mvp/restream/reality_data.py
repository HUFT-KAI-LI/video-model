"""Image decoding and deterministic identities shared by filtering and feature caches."""
import hashlib
import json
from pathlib import Path
import av
import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_reference(reference, size=None, return_time=False, latest=False):
    """Decode one video frame near a metadata timestamp; never feed time to the model.

    Default: first frame at/after ``reference["time"]``. With ``latest=True``
    return the last frame whose time is at/before the timestamp (plus a small
    decode tolerance), which strict-online analysis uses so that the final
    histogram frame cannot start after the visible cutoff.
    """
    with av.open(reference["video"]) as container:
        stream = container.streams.video[0]
        origin = float((stream.start_time or 0) * stream.time_base)
        # Seeking keeps feature preparation practical for long sources.
        container.seek(int((reference["time"] + origin) / stream.time_base), stream=stream, backward=True)
        match = None
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            sampled_time = float(frame.time - origin)
            if latest:
                if sampled_time <= reference["time"] + 1e-6:
                    match = (frame, sampled_time)
                else:
                    break
            elif sampled_time + 1e-6 >= reference["time"]:
                match = (frame, sampled_time)
                break
        if match is None:
            raise ValueError(f"Reference timestamp cannot be decoded: {reference['video']} @ {reference['time']}")
        frame, sampled_time = match
        if size:
            frame = frame.reformat(width=size, height=size, format="rgb24")
        rgb = frame.to_ndarray(format="rgb24")
        return (rgb, sampled_time) if return_time else rgb


def histogram(rgb):
    bins = rgb.astype(np.int32) // 32
    ids = bins[..., 0] * 64 + bins[..., 1] * 8 + bins[..., 2]
    counts = np.bincount(ids.ravel(), minlength=512).astype(np.float32)
    return counts / counts.sum()


def scene_similarity(left, right):
    # Bhattacharyya coefficient, not a proof of semantic scene identity.
    return float(np.sqrt(left * right).sum())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
