#!/usr/bin/env python3
"""Build the frozen D2 binary history-subset manifest."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.history_subsets import CONDITIONS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edits", nargs="+", default=[
        "dress_red_to_blue", "jacket_green_to_yellow",
        "lighting_warm_to_cool", "lighting_darker"])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/history_subset_interaction_manifest.json")
    args = parser.parse_args()
    cases = [{"edit": edit, "seed": 404, "target_chunk": 4, "condition": condition}
             for edit in args.edits for condition in CONDITIONS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": 1, "experiment": "history_subset_interaction_d2",
                                       "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
