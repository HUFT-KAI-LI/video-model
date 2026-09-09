"""Conservative all-frame histogram/black-frame/transition filter; no training."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import read_manifest
from restream.reality_data import canonical_hash, histogram, write_json
from restream.runtime import read_config


def scan(row, config):
    shots, start, previous, previous_hist, last = [], None, None, None, None
    cuts, black, transitions, decoded = 0, 0, 0, 0
    try:
        with av.open(row["video"]) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            origin = float((stream.start_time or 0) * stream.time_base)
            for frame in container.decode(stream):
                if frame.time is None:
                    continue
                at = float(frame.time - origin)
                rgb = frame.reformat(width=64, height=64, format="rgb24").to_ndarray()
                current = histogram(rgb)
                dark = float((rgb.max(-1) < config["black_level"]).mean()) >= config["black_fraction"]
                cut = previous is not None and float(np.abs(current - previous_hist).sum() / 2) > config["histogram_cut"]
                transition = previous is not None and float(np.abs(rgb.astype(float) - previous).mean() / 255) > config["pixel_jump"]
                if dark or cut or transition:
                    if start is not None and last is not None and last > start:
                        shots.append({"start": start, "end": last})
                    start = None
                if not dark and start is None:
                    start = at
                cuts += int(cut)
                black += int(dark)
                transitions += int(transition)
                previous, previous_hist, last = rgb.astype(float), current, at
                decoded += 1
            if start is not None and last > start:
                shots.append({"start": start, "end": last})
        return {"source_id": row["source_id"], "video": row["video"], "video_sha256": row["sha256"],
                "filter_hash": canonical_hash(config), "shots": shots, "decoded_frames": decoded,
                "hard_cuts": cuts, "black_frames": black, "transitions": transitions, "error": None}
    except (av.error.FFmpegError, ValueError, OSError) as error:
        return {"source_id": row["source_id"], "shots": [], "error": str(error)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data/reality_shots.jsonl")
    args = parser.parse_args()
    config = read_config(args.config)["reality_memory"]["filter"]
    rows = read_manifest(ROOT / "data/train.jsonl") + read_manifest(ROOT / "data/val.jsonl")
    rows = list({row["source_id"]: row for row in rows}.values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    records = []
    with ThreadPoolExecutor(max_workers=config["workers"]) as pool, temporary.open("w") as output:
        for i, record in enumerate(pool.map(lambda row: scan(row, config), rows)):
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                print(f"Scanned {i + 1}/{len(rows)} videos", flush=True)
    temporary.replace(args.output)
    write_json(args.output.with_suffix(".stats.json"), {
        "sources": len(records), "errors": sum(r["error"] is not None for r in records), "config": config,
        "note": "All decoded frames checked. Histogram heuristics do not certify same-world identity or human visual quality."})


if __name__ == "__main__":
    main()
