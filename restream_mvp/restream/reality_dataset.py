"""Reuse the verified target video decoder and add cached, variable-count references."""
import torch
from torch.utils.data._utils.collate import default_collate
from .dataset import VideoDataset


class RealityDataset(VideoDataset):
    def __init__(self, manifest, cache, frames=57, height=256, width=432, fps=16, preflight=True):
        super().__init__(manifest, frames, height, width, fps)
        self.cache = cache
        for row in self.rows:
            if row["target_start"] != row["window_start"] or row["target_sec"] != row["window_sec"]:
                raise ValueError("Target and reused decoder window disagree")
            for ref in row["references"]:
                if ref["split"] != row["split"]:
                    raise ValueError("Cross-split reference leakage")
                wrong = row["reference_kind"] == "wrong"
                if (ref["source_id"] != row["source_id"]) != wrong:
                    raise ValueError("Reference source does not match its training label")
                if preflight and not cache.path(ref).exists():
                    raise FileNotFoundError(f"Uncached reference in sample {row['sample_id']}; run feature preparation")

    def reference_features(self, references):
        if not references:
            return torch.empty(0, self.cache.tokens, self.cache.channels)
        return torch.stack([self.cache.read(ref) for ref in references])

    def __getitem__(self, index):
        sample, row = super().__getitem__(index), self.rows[index]
        features = self.reference_features(row["references"])
        return {**sample, "features": features, "reference_mask": torch.ones(features.shape[0], dtype=torch.bool),
                "wrong_reference": row["reference_kind"] == "wrong", "sample_id": row["sample_id"],
                "reference_kind": row["reference_kind"]}


def collate_reality(samples):
    count = max(1, max(sample["features"].shape[0] for sample in samples))
    padded = []
    for sample in samples:
        features = sample["features"]
        out = features.new_zeros(count, *features.shape[1:])
        mask = torch.zeros(count, dtype=torch.bool)
        out[:len(features)], mask[:len(features)] = features, sample["reference_mask"]
        padded.append({**sample, "features": out, "reference_mask": mask})
    return default_collate(padded)
