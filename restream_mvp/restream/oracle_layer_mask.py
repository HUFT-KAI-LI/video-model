"""Frozen M0/M1 oracle layer-release optimization protocol."""
from __future__ import annotations

import math

from . import edit_experiment as ex


PROTOCOL = "oracle_layer_release_m0m1_v1"
BASELINES = ("full", "global_.5", "current_only")


def manifest_groups(config, manifest):
    if manifest.get("schema") != 1 or manifest.get("experiment") != "oracle_layer_release_m0m1":
        raise ValueError("Use the M0/M1 oracle layer-mask manifest schema")
    prompts = {prompt["id"]: prompt for prompt in ex.prompt_cases(config)}
    groups, seen = [], set()
    for entry in manifest.get("cases", []):
        if set(entry) != {"edit", "seed", "target_chunk"}:
            raise ValueError("oracle cases must contain only edit, seed, and target_chunk")
        key = (entry["edit"], entry["seed"], entry["target_chunk"])
        if key in seen:
            raise ValueError(f"duplicate oracle unit {key}")
        seen.add(key)
        prompt = prompts.get(entry["edit"])
        if prompt is None:
            raise ValueError(f"unknown edit {entry['edit']}")
        if type(entry["seed"]) is not int or entry["target_chunk"] != 4:
            raise ValueError("oracle units require integer seeds and target_chunk=4")
        groups.append({"prompt_id": prompt["id"], "prompt_index": prompt["index"],
                       "base_prompt": prompt["base"], "edit_prompt": prompt["edit"],
                       "probe": prompt["probe"], "evidence": prompt["evidence"],
                       "seed": entry["seed"], "target_chunk": 4})
    return groups


def validate_plan(groups, plan):
    manifest = plan["manifest"]
    expected = {(edit, seed, 4) for edit in manifest["edits"] for seed in manifest["seeds"]}
    actual = {(group["prompt_id"], group["seed"], group["target_chunk"]) for group in groups}
    if actual != expected or len(groups) != manifest["expected_units"]:
        raise ValueError("oracle edit/seed grid differs from frozen plan")
    optimization = plan["optimization"]
    if optimization["algorithm"] != "SPSA" or optimization["layer_count"] != 30:
        raise ValueError("oracle optimizer or layer count differs from frozen plan")
    lambdas = optimization["lambdas"]
    if len(lambdas) != 3 or any(not math.isfinite(x) or x < 0 for x in lambdas):
        raise ValueError("oracle plan requires three finite nonnegative lambdas")


def oracle_name(drift_lambda):
    return f"oracle_lambda_{float(drift_lambda):g}"
