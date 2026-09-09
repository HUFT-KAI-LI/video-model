"""Reuse the verified target video decoder and add cached, variable-count references."""
import math
import torch
from torch.utils.data._utils.collate import default_collate
from .dataset import VideoDataset


class RealityDataset(VideoDataset):
    def __init__(self, manifest, cache, frames=57, height=256, width=432, fps=16, preflight=True,
                 selection_protocol=None, selection_config_hash=None):
        super().__init__(manifest, frames, height, width, fps)
        self.cache = cache
        self.selection_protocol = selection_protocol
        self.selection_config_hash = selection_config_hash
        for row in self.rows:
            actual_protocol = row.get("selection_protocol", "offline_target_filtered")
            if selection_protocol is not None and actual_protocol != selection_protocol:
                raise ValueError(f"Manifest selection protocol {actual_protocol!r} differs from requested {selection_protocol!r}; rebuild manifest")
            if selection_config_hash is not None and "selection_config_hash" in row and row["selection_config_hash"] != selection_config_hash:
                raise ValueError(f"Manifest selection_config_hash {row['selection_config_hash']!r} differs from the current selection configuration {selection_config_hash!r}; rebuild manifest")
            if actual_protocol == "strict_online":
                self._validate_strict_online_row(row)
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

    def _validate_strict_online_row(self, row):
        """A strict-online row must be causally closed on every positive pool.

        The reference sets (async + aligned) decide what a paired probe can read,
        so the visible-until bound is checked on the full pools, not only on the
        mixture-selected ``row["references"]`` slice.
        """
        visible_until = row.get("visible_until")
        if not isinstance(visible_until, (int, float)) or not math.isfinite(visible_until):
            raise ValueError("strict_online rows require a finite visible_until")
        if row.get("selection_schema") is None:
            raise ValueError("strict_online rows require the selection schema field")
        for kind in ("async", "aligned"):
            for ref in row.get("reference_sets", {}).get(kind, ()):
                if ref.get("split") != row["split"]:
                    raise ValueError("Cross-split reference leakage in a strict-online pool")
                if not isinstance(ref.get("time"), (int, float)) or ref["time"] > visible_until + 1e-6:
                    raise ValueError(f"strict_online {kind} reference at {ref.get('time')} exceeds visible_until {visible_until}")

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
