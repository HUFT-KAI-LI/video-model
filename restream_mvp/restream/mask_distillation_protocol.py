"""Frozen manifests and checks for M1-A mask distillation."""
from __future__ import annotations

from . import edit_experiment as ex
from .mask_distillation import PROTOCOL


EDITS = ("dress_red_to_blue", "jacket_green_to_yellow",
         "lighting_warm_to_cool", "lighting_darker")
TEACHER_SEEDS = (606, 707, 801, 802, 803, 804, 805, 806, 807, 808)
NEW_TEACHER_SEEDS = TEACHER_SEEDS[2:]
HELDOUT_SEEDS = (1001, 1002)
FINAL_CONDITIONS = ("full", "global_.5", "current_only", "prompt_only",
                    "prompt_state", "oracle")
M1B_NEW_TEACHER_SEEDS = tuple(range(1101, 1141))
M1B_HELDOUT_SEEDS = (2001, 2002, 2003, 2004)
M1B_FINAL_CONDITIONS = ("full", "global_.5", "prompt_only", "prompt_state", "oracle")


def groups_from_manifest(config, manifest, experiment, allowed_seeds):
    if manifest.get("schema") != 1 or manifest.get("experiment") != experiment:
        raise ValueError(f"wrong {experiment} manifest schema")
    prompts = {prompt["id"]: prompt for prompt in ex.prompt_cases(config)}
    groups, seen = [], set()
    for entry in manifest.get("cases", []):
        if set(entry) != {"edit", "seed", "target_chunk"}:
            raise ValueError("cases must contain only edit, seed, and target_chunk")
        key = (entry["edit"], entry["seed"], entry["target_chunk"])
        if key in seen or entry["edit"] not in EDITS or entry["seed"] not in allowed_seeds or entry["target_chunk"] != 4:
            raise ValueError(f"invalid or duplicate unit {key}")
        seen.add(key)
        prompt = prompts[entry["edit"]]
        groups.append({"prompt_id": prompt["id"], "prompt_index": prompt["index"],
                       "base_prompt": prompt["base"], "edit_prompt": prompt["edit"],
                       "probe": prompt["probe"], "evidence": prompt["evidence"],
                       "seed": entry["seed"], "target_chunk": 4})
    expected = {(edit, seed, 4) for edit in EDITS for seed in allowed_seeds}
    if seen != expected:
        raise ValueError(f"{experiment} grid is incomplete")
    return groups
