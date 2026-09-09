"""Verify keyframe seek returns exactly the same target pixels on real late windows."""
from pathlib import Path
import sys
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import VideoDataset
from restream.reality_data import write_json


if __name__ == "__main__":
    seek = VideoDataset(ROOT / "data/reality_train.jsonl", seek=True)
    sequential = VideoDataset(ROOT / "data/reality_train.jsonl", seek=False)
    ordered = sorted(range(len(seek)), key=lambda i: seek.rows[i]["window_start"])
    indices = [ordered[int((len(ordered) - 1) * fraction)] for fraction in (0, .25, .5, .75, .9, .95, .99, 1)]
    cases = []
    for index in indices:
        started = time.perf_counter()
        old = sequential[index]["pixels"]
        sequential_time = time.perf_counter() - started
        started = time.perf_counter()
        new = seek[index]["pixels"]
        seek_time = time.perf_counter() - started
        if not torch.equal(old, new):
            raise RuntimeError(f"Seek changed sampled pixels: {seek.rows[index]['sample_id']}")
        record = {"sample_id": seek.rows[index]["sample_id"], "target_start": seek.rows[index]["window_start"],
                  "sequential_seconds": sequential_time, "seek_seconds": seek_time, "pixels_exact": True}
        cases.append(record)
        print(record, flush=True)
    write_json(ROOT / "validation/reality_memory/overfit_review/target_seek.json", {"status": "passed", "cases": cases,
               "note": "Sequential decode runs first; timings include filesystem-cache effects, not a randomized benchmark."})
