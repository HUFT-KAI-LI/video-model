#!/usr/bin/env python3
"""Build M1-A new-teacher and held-out manifests."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EDITS = ["dress_red_to_blue", "jacket_green_to_yellow",
         "lighting_warm_to_cool", "lighting_darker"]


def write(name, experiment, seeds):
    cases = [{"edit": edit, "seed": seed, "target_chunk": 4}
             for edit in EDITS for seed in seeds]
    (ROOT / "validation" / name).write_text(json.dumps(
        {"schema": 1, "experiment": experiment, "cases": cases}, indent=2) + "\n")


if __name__ == "__main__":
    write("mask_distillation_teacher_manifest.json", "mask_distillation_teacher_m1a", range(801, 809))
    write("mask_distillation_heldout_manifest.json", "mask_distillation_heldout_m1a", (1001, 1002))
    write("mask_distillation_m1b_teacher_manifest.json", "mask_distillation_teacher_m1b", range(1101, 1141))
    write("mask_distillation_m1b_heldout_manifest.json", "mask_distillation_heldout_m1b", (2001, 2002, 2003, 2004))
