import json
import math
from pathlib import Path
import av
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class VideoDataset(Dataset):
    def __init__(self, manifest, frames=57, height=256, width=432, fps=16):
        self.rows = read_manifest(manifest)
        if not self.rows:
            raise ValueError("Empty dataset")
        if frames < 9 or min(height, width) <= 0 or (frames - 1) % 4 or ((frames - 1) // 4 + 1) % 3 or height % 16 or width % 16:
            raise ValueError("Expected 4k+1 pixels, latent frames multiple of 3, spatial multiple of 16")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive")
        self.frames, self.height, self.width, self.fps = frames, height, width, float(fps)
        window_sec = (frames - 1) / self.fps
        if any(not math.isclose(float(row["window_sec"]), window_sec, abs_tol=1e-6) for row in self.rows):
            raise ValueError("Manifest window_sec differs from (frames - 1) / fps; rebuild the manifest with the current config")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        start = float(row["window_start"])
        targets = start + np.arange(self.frames, dtype=np.float64) / self.fps
        decoded, cursor = [], 0
        with av.open(row["video"]) as container:
            stream = container.streams.video[0]
            origin = float((stream.start_time or 0) * stream.time_base)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                at = frame.time - origin
                if cursor < self.frames and at + 1e-6 >= targets[cursor]:
                    tensor = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).float()
                    scale = max(self.height / tensor.shape[1], self.width / tensor.shape[2])
                    h, w = max(self.height, round(tensor.shape[1] * scale)), max(self.width, round(tensor.shape[2] * scale))
                    tensor = F.interpolate(tensor[None], (h, w), mode="bilinear", align_corners=False)[0]
                    y, x = (h - self.height) // 2, (w - self.width) // 2
                    tensor = tensor[:, y:y + self.height, x:x + self.width] / 127.5 - 1
                    while cursor < self.frames and at + 1e-6 >= targets[cursor]:
                        decoded.append(tensor)
                        cursor += 1
                if cursor == self.frames:
                    break
        if cursor != self.frames:
            raise ValueError(f"Video too short for manifest window: {row['video']}")
        return {"pixels": torch.stack(decoded, dim=1), "caption": row["caption"],
                "source_id": row["source_id"], "window_sec": row["window_sec"],
                "anchor_sec": row["anchor_sec"][0]}
