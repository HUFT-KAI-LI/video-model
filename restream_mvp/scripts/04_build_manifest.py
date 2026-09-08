import argparse
import json
import random
import statistics
from pathlib import Path


def build(raw, output, seed=42):
    rows = [json.loads(x) for x in raw.read_text().splitlines() if x.strip()]
    groups = {}
    for row in rows:
        groups.setdefault(str(row["source_id"]), []).append(row)
    ids = sorted(groups)
    random.Random(seed).shuffle(ids)
    if len(ids) < 2:
        raise ValueError("Need at least two distinct source videos")
    val_ids = set(ids[:max(1, round(len(ids) * .1))])
    splits = {"train": [], "val": []}
    for source in ids:
        rng = random.Random(f"{seed}:{source}")
        for row in groups[source]:
            duration = float(row["duration"])
            if duration < 4:
                continue
            length = min(duration - .2, 3.5)
            split = "val" if source in val_ids else "train"
            splits[split].append({**row, "source_id": source, "split": split,
                                  "window_start": rng.uniform(0, duration - .2 - length),
                                  "window_sec": length,
                                  "anchor_sec": [rng.uniform(.30, .40) * length]})
    output.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        if not values:
            raise ValueError(f"No usable {name} videos")
        (output / f"{name}.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))
    usable = splits["train"] + splits["val"]
    durations = [r["duration"] for r in usable]
    stats = {"download_gb": sum(Path(v).stat().st_size for v in {r["video"] for r in rows}) / 1e9,
             "total_source_videos": len(groups), "usable_videos": len(usable),
             "train_sources": len({r["source_id"] for r in splits["train"]}),
             "val_sources": len({r["source_id"] for r in splits["val"]}),
             "mean_duration_sec": statistics.mean(durations), "median_duration_sec": statistics.median(durations),
             "min_duration_sec": min(durations), "max_duration_sec": max(durations), "seed": seed}
    stats["minimum_count_met"] = stats["train_sources"] >= 300 and stats["val_sources"] >= 40
    (output / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    build(a.raw, a.output, a.seed)
