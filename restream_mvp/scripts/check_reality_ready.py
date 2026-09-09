"""Audit split boundaries, temporal exclusions, cached tensors and baseline preservation."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import read_manifest
from restream.reality_data import write_json
from restream.reality_runtime import make_cache, make_dataset, read_reality_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/readiness.json")
    args = parser.parse_args()
    config = read_reality_config(args.config)
    cache = make_cache(config)
    rows = {split: read_manifest(ROOT / config["data"][f"{split}_manifest"]) for split in ("train", "val")}
    sources = {split: {row["source_id"] for row in values} for split, values in rows.items()}
    contents = {split: {row["sha256"] for row in values} for split, values in rows.items()}
    assert not sources["train"] & sources["val"]
    assert not contents["train"] & contents["val"]
    checked, originals = {}, {}
    for split, values in rows.items():
        originals[split] = hashlib.sha256((ROOT / f"data/{split}.jsonl").read_bytes()).hexdigest()
        dataset = make_dataset(config, split, cache)
        assert len(dataset) == len(values) > 0
        for row in values:
            with Path(row["video"]).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
            assert row["target_sec"] == (config["data"]["frames"] - 1) / config["data"]["fps"]
            assert row["shot"]["start"] < row["target_start"]
            assert row["shot"]["end"] > row["target_start"] + row["target_sec"]
            for kind, pool in row["reference_sets"].items():
                assert len({(ref["video_sha256"], ref["time"]) for ref in pool}) == len(pool)
                for ref in pool:
                    assert ref["source_id"] in sources[split] and ref["split"] == split
                    if kind == "wrong":
                        assert ref["source_id"] != row["source_id"] and ref["video_sha256"] != row["sha256"]
                    else:
                        assert ref["source_id"] == row["source_id"] and ref["shot_id"] == row["shot_id"]
                    if kind == "async":
                        gap = config["reality_memory"]["references"]["min_gap_sec"]
                        assert ref["time"] <= row["target_start"] - gap + 1e-6 or (
                            config["reality_memory"]["references"]["async_direction"] == "both" and
                            ref["time"] >= row["target_start"] + row["target_sec"] + gap - 1e-6)
                    if split == "val" or ref in row["references"]:
                        key = cache.key(ref)
                        if key not in checked:
                            cache.read(ref)
                            checked[key] = hashlib.sha256(cache.path(ref).read_bytes()).hexdigest()
        # Decode representative rows of all four types using the reused decoder.
        for kind in ("async", "aligned", "none", "wrong"):
            index = next(i for i, row in enumerate(values) if row["reference_kind"] == kind)
            sample = dataset[index]
            assert sample["pixels"].shape == (3, config["data"]["frames"], config["data"]["height"], config["data"]["width"])
    report = {"status": "passed", "train_sources": len(rows["train"]), "val_sources": len(rows["val"]),
              "cross_split_source_overlap": 0, "cross_split_video_content_overlap": 0,
              "checked_feature_files": len(checked), "features_digest": hashlib.sha256(str(sorted(checked.items())).encode()).hexdigest(),
              "original_manifests_sha256": originals, "encoder": cache.identity, "optimizer_steps": 0}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
