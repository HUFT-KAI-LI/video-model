#!/usr/bin/env python3
"""Create a reproducible history-release sweep manifest.

The manifest is consumed by the existing edit experiment runner; generation is
kept separate so every case records its gate, target chunk and seed; P0/P1 run internally.
"""
import argparse, json
from pathlib import Path

DEFAULT_GATES = (1.0, .75, .5, .25, 0.0)

def build_cases(edits, seeds, chunks, gates):
    return [{"edit": e, "seed": int(s), "target_chunk": int(c),
             "history_gate": float(g)}
            for e in edits for s in seeds for c in chunks for g in gates]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edits", nargs="+", default=["dress_red_to_blue", "jacket_green_to_yellow", "lighting_warm_to_cool", "lighting_darker"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[101, 202])
    ap.add_argument("--chunks", nargs="+", type=int, default=[0, 1, 4])
    ap.add_argument("--gates", nargs="+", type=float, default=DEFAULT_GATES)
    ap.add_argument("--output", type=Path, default=Path("validation/history_release_sweep_manifest.json"))
    a = ap.parse_args(); cases = build_cases(a.edits, a.seeds, a.chunks, a.gates)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"schema": 2, "cases": cases}, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {a.output}")

if __name__ == "__main__": main()
