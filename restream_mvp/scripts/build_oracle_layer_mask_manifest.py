#!/usr/bin/env python3
"""Build the frozen M0/M1 oracle layer-mask unit manifest."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITS = ["dress_red_to_blue", "jacket_green_to_yellow",
         "lighting_warm_to_cool", "lighting_darker"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/oracle_layer_mask_manifest.json")
    args = parser.parse_args()
    cases = [{"edit": edit, "seed": seed, "target_chunk": 4}
             for edit in EDITS for seed in (606, 707)]
    args.output.write_text(json.dumps({"schema": 1,
                                      "experiment": "oracle_layer_release_m0m1",
                                      "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    main()
