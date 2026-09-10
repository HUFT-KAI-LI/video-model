"""Reuse the verified target video decoder and add cached, variable-count references."""
import math
import torch
from torch.utils.data._utils.collate import default_collate
from .dataset import VideoDataset
from .reality_selection import SELECTION_IDENTITY_SCHEMA
from .reality_temporal import latent_prefix_boundary_index, prefix_visible_seconds


class RealityDataset(VideoDataset):
    def __init__(self, manifest, cache, frames=57, height=256, width=432, fps=16, preflight=True,
                 selection_protocol=None, selection_config_hash=None, prefix_latents=None,
                 temporal_sampling=None, allow_legacy_offline_manifest=False):
        super().__init__(manifest, frames, height, width, fps,
                         temporal_sampling=temporal_sampling or "first_at_or_after")
        self.cache = cache
        self.selection_protocol = selection_protocol
        self.selection_config_hash = selection_config_hash
        self.prefix_latents = prefix_latents
        self.allow_legacy_offline_manifest = allow_legacy_offline_manifest
        for row in self.rows:
            actual_protocol = row.get("selection_protocol", "offline_target_filtered")
            if selection_protocol is not None and actual_protocol != selection_protocol:
                raise ValueError(f"Manifest selection protocol {actual_protocol!r} differs from requested {selection_protocol!r}; rebuild manifest")
            if temporal_sampling is not None:
                row_sampling = row.get("temporal_sampling")
                if row_sampling is not None and row_sampling != temporal_sampling:
                    raise ValueError(f"Manifest temporal_sampling {row_sampling!r} differs from requested {temporal_sampling!r}; rebuild manifest")
            if actual_protocol == "strict_online":
                self._validate_strict_online_row(row)
            else:
                self._validate_offline_row(row)
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

    def _validate_offline_row(self, row):
        """Legacy offline rows (no selection identity) need an explicit opt-in."""
        has_identity = "selection_schema" in row or "selection_config_hash" in row
        if not has_identity:
            if not self.allow_legacy_offline_manifest:
                raise ValueError("Legacy offline manifest rows carry no selection schema/hash; set references.allow_legacy_offline_manifest: true for the old manifest or rebuild it")
            return
        if row.get("selection_schema") != SELECTION_IDENTITY_SCHEMA:
            raise ValueError(f"Manifest selection_schema {row.get('selection_schema')!r} differs from {SELECTION_IDENTITY_SCHEMA}; rebuild manifest")
        if self.selection_config_hash is None:
            raise ValueError("Manifest carries a selection hash but no expected selection_config_hash was provided")
        if row.get("selection_config_hash") != self.selection_config_hash:
            raise ValueError(f"Manifest selection_config_hash {row.get('selection_config_hash')!r} differs from the current selection configuration {self.selection_config_hash!r}; rebuild manifest")

    def _validate_strict_online_row(self, row):
        """A strict-online row must be causally closed on every positive pool.

        The reference sets (async + aligned) decide what a paired probe can read,
        so the visible-until bound is checked on the full pools, not only on the
        mixture-selected ``row["references"]`` slice. The identity fields are
        mandatory: a missing hash must not be treated as legacy compatibility.
        """
        if row.get("selection_schema") != SELECTION_IDENTITY_SCHEMA:
            raise ValueError(f"strict_online rows require selection_schema {SELECTION_IDENTITY_SCHEMA}, got {row.get('selection_schema')!r}")
        if self.selection_config_hash is None:
            raise ValueError("strict_online rows require an expected selection_config_hash")
        if row.get("selection_config_hash") != self.selection_config_hash:
            raise ValueError("strict_online row selection_config_hash differs from the current selection configuration; rebuild manifest")
        if row.get("temporal_sampling") != "causal_previous" or self.temporal_sampling != "causal_previous":
            raise ValueError("strict_online rows require temporal_sampling=causal_previous in both manifest and Dataset")
        visible_until = row.get("visible_until")
        if not isinstance(visible_until, (int, float)) or not math.isfinite(visible_until):
            raise ValueError("strict_online rows require a finite visible_until")
        if self.prefix_latents is None:
            raise ValueError("strict_online rows require prefix_latents to bound visible_until")
        start = float(row["target_start"])
        arrival = prefix_visible_seconds(self.prefix_latents, self.fps)
        if not start - 1e-6 <= visible_until <= start + arrival + 1e-6:
            raise ValueError(f"visible_until {visible_until} outside [target_start, target_start+arrival] = [{start}, {start + arrival}]")
        for kind in ("async", "aligned"):
            for ref in row.get("reference_sets", {}).get(kind, ()):
                if ref.get("split") != row["split"]:
                    raise ValueError("Cross-split reference leakage in a strict-online pool")
                if ref.get("source_id") != row["source_id"]:
                    raise ValueError("Foreign-source reference in a strict-online positive pool")
                if ref.get("shot_id") != row["shot_id"]:
                    raise ValueError("Cross-shot reference in a strict-online positive pool")
                if not isinstance(ref.get("time"), (int, float)) or ref["time"] > visible_until + 1e-6:
                    raise ValueError(f"strict_online {kind} reference at {ref.get('time')} exceeds visible_until {visible_until}")

    def _assert_prefix_matches_visible_until(self, row, sample):
        """Runtime causality: the decoded prefix must end exactly at the manifest
        visible_until under causal_previous, so a manifest cannot claim a boundary
        the Dataset does not actually consume."""
        times = sample["sampled_times"]
        boundary = latent_prefix_boundary_index(self.prefix_latents)
        if times.numel() <= boundary:
            raise ValueError("Sampled window shorter than the strict prefix boundary")
        end = float(times[boundary])
        visible_until = float(row["visible_until"])
        if abs(end - visible_until) > 1e-5:
            raise ValueError(f"Dataset prefix end {end} differs from manifest visible_until {visible_until}")
        if float(times[:boundary + 1].max()) > visible_until + 1e-5:
            raise ValueError("A prefix frame starts after the manifest visible_until")

    def reference_features(self, references):
        if not references:
            return torch.empty(0, self.cache.tokens, self.cache.channels)
        return torch.stack([self.cache.read(ref) for ref in references])

    def __getitem__(self, index):
        sample, row = super().__getitem__(index), self.rows[index]
        if row.get("selection_protocol", "offline_target_filtered") == "strict_online":
            self._assert_prefix_matches_visible_until(row, sample)
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
