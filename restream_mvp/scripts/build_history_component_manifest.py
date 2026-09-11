#!/usr/bin/env python3
"""Build the exploratory D1 sink/old/recent component-screen manifest."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restream.history_components import CONDITIONS


def build_cases(edits, seeds, chunks, conditions):
    return [{"edit": edit, "seed": int(seed), "target_chunk": int(chunk),
             "condition": condition}
            for edit in edits for seed in seeds for chunk in chunks for condition in conditions]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edits", nargs="+", default=[
        "dress_red_to_blue", "jacket_green_to_yellow",
        "lighting_warm_to_cool", "lighting_darker"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[303])
    parser.add_argument("--chunks", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--output", type=Path,
                        default=Path("validation/history_component_screen_manifest.json"))
    args = parser.parse_args()
    unknown = set(args.conditions) - set(CONDITIONS)
    if unknown:
        parser.error(f"unknown conditions: {sorted(unknown)}")
    cases = build_cases(args.edits, args.seeds, args.chunks, args.conditions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": 1, "experiment": "history_component_screen_d1",
                                      "cases": cases}, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
