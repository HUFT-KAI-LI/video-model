"""Precompute frozen reference features. Never runs the video generator or an optimizer."""
import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import read_manifest
from restream.reality_data import read_reference, write_json
from restream.reality_encoder import RealityEncoder
from restream.reality_runtime import read_reality_config, make_cache
from restream.reality_selection import (GLOBAL_MEAN_SCHEMA, manifest_digest, reference_keys_digest,
                                        selection_config_hash, unique_train_reference_pool)


def global_mean_payload(rows, cache, config):
    """Fixed train-split mean over the deduplicated positive reference pools.

    The pool is the union of every row's ``reference_sets.async`` and
    ``reference_sets.aligned``, deduplicated by content key. It is therefore
    independent of the training mixture (no-memory/wrong probabilities, sampled
    K per row, donor references re-used across rows).
    """
    unique = unique_train_reference_pool(rows, cache)
    keys = sorted(unique)
    manifest = ROOT / config["data"]["train_manifest"]
    references = config["reality_memory"]["references"]
    if not keys:
        raise ValueError("Train manifest has no async/aligned reference pool for a global mean")
    features = torch.stack([cache.read(unique[key]) for key in keys]).mean(0).float()
    return {
        "schema": GLOBAL_MEAN_SCHEMA,
        "features": features,
        "source_split": "train",
        "unique_reference_count": len(keys),
        "cache_identity": cache.identity,
        "tokens": cache.tokens,
        "channels": cache.channels,
        "selection_protocol": references.get("selection_protocol", "offline_target_filtered"),
        "train_manifest_sha256": manifest_digest(manifest),
        "reference_keys_sha256": reference_keys_digest(unique),
        "selection_config_hash": selection_config_hash(config),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overfit-samples", type=int, default=0, help="Also cache all control pools for a balanced training subset")
    args = parser.parse_args()
    config = read_reality_config(args.config)
    memory = config["reality_memory"]
    cache = make_cache(config)
    refs = {}
    for split in ("train", "val"):
        rows = read_manifest(ROOT / config["data"][f"{split}_manifest"])
        overfit = set()
        if split == "train" and args.overfit_samples:
            from types import SimpleNamespace
            from train_reality_memory import overfit_indices
            overfit = set(overfit_indices(SimpleNamespace(rows=rows), args.overfit_samples))
        for index, row in enumerate(rows):
            # Cache the mixture-selected slice plus every async/aligned pool frame
            # (the deduplicated global mean reads the full pools), and keep the
            # wrong-source pools for val and for the balanced overfit subset.
            extra = ("wrong",) if (split == "val" or index in overfit) else ()
            for ref in row.get("references", ()):
                refs[cache.key(ref)] = ref
            for kind in ("async", "aligned") + extra:
                for ref in row.get("reference_sets", {}).get(kind, ()):
                    refs[cache.key(ref)] = ref
    encoder = RealityEncoder(ROOT / memory["encoder"]["path"], memory["projector"]["memory_dim"],
                             memory["projector"]["num_memory_tokens"], memory["encoder"]["image_size"]).to(args.device).eval()
    written, reused = 0, 0
    for i, ref in enumerate(refs.values()):
        if cache.path(ref).exists():
            cache.read(ref)  # Shape, finite values and identity check on reuse.
            reused += 1
        else:
            pixels = torch.from_numpy(read_reference(ref)).permute(2, 0, 1).float() / 255
            features = encoder.encode_visual(pixels[None, None])[0, 0]
            cache.write(ref, features)
            written += 1
        if (i + 1) % 100 == 0:
            print(f"Features {i + 1}/{len(refs)}", flush=True)
    report = {"unique_references": len(refs), "written": written, "reused": reused, "encoder": cache.identity,
              "shape_per_reference": [cache.tokens, cache.channels], "optimizer_steps": 0}
    train_rows = read_manifest(ROOT / config["data"]["train_manifest"])
    global_mean_path = ROOT / memory["references"].get("global_constant_features", "data/reality_global_mean_features.pt")
    payload = global_mean_payload(train_rows, cache, config)
    global_mean_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = global_mean_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(global_mean_path)
    report["global_constant"] = {"path": str(global_mean_path.relative_to(ROOT)), "source_split": "train",
                                 "schema": payload["schema"], "unique_reference_count": payload["unique_reference_count"],
                                 "tokens": payload["tokens"], "channels": payload["channels"],
                                 "selection_protocol": payload["selection_protocol"],
                                 "train_manifest_sha256": payload["train_manifest_sha256"],
                                 "reference_keys_sha256": payload["reference_keys_sha256"],
                                 "selection_config_hash": payload["selection_config_hash"]}
    if args.overfit_samples:
        report["overfit_samples"] = args.overfit_samples
    write_json(ROOT / ("data/reality_overfit_feature_stats.json" if args.overfit_samples else "data/reality_feature_stats.json"), report)
    print(report)


if __name__ == "__main__":
    main()