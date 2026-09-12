"""Frozen D3 history attention-path intervention definitions."""
import math

from . import edit_experiment as ex


PROTOCOL = "fixed_history_attention_paths_d3_v1"
CONDITIONS = {
    "full": {"kind": "path", "score_gate": 1.0, "value_gate": 1.0},
    "global_.5": {"kind": "global", "gate": 0.5},
    "current_only": {"kind": "global", "gate": 0.0},
    "score_.5": {"kind": "path", "score_gate": 0.5, "value_gate": 1.0},
    "value_.5": {"kind": "path", "score_gate": 1.0, "value_gate": 0.5},
    "value_0": {"kind": "path", "score_gate": 1.0, "value_gate": 0.0},
    "score_.5_value_.5": {"kind": "path", "score_gate": 0.5, "value_gate": 0.5},
}
CONTROLS = {"score_0": {"kind": "path", "score_gate": 0.0, "value_gate": 1.0}}


def condition_spec(name):
    if name not in CONDITIONS and name not in CONTROLS:
        raise ValueError(f"Unknown D3 condition {name}")
    spec = (CONDITIONS | CONTROLS)[name]
    if spec["kind"] == "global":
        return {"kind": "global", "gate": float(spec["gate"])}
    return {"kind": "path", "score_gate": float(spec["score_gate"]),
            "value_gate": float(spec["value_gate"])}


def manifest_groups(config, manifest):
    if manifest.get("schema") != 1 or manifest.get("experiment") != "history_attention_path_d3":
        raise ValueError("Use the D3 history-path manifest schema")
    prompts = {prompt["id"]: prompt for prompt in ex.prompt_cases(config)}
    groups, seen = {}, set()
    for entry in manifest.get("cases", []):
        if set(entry) != {"edit", "seed", "target_chunk", "condition"}:
            raise ValueError("D3 cases must contain only edit, seed, target_chunk, and condition")
        prompt = prompts.get(entry["edit"])
        if prompt is None:
            raise ValueError(f"Unknown edit {entry['edit']}")
        seed, target, condition = entry["seed"], entry["target_chunk"], entry["condition"]
        if type(seed) is not int or type(target) is not int or target < 1:
            raise ValueError("seed and target_chunk must be positive-grid integers")
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown D3 main-grid condition {condition}")
        spec = condition_spec(condition)
        values = ([spec["gate"]] if spec["kind"] == "global" else
                  [spec["score_gate"], spec["value_gate"]])
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("D3 path gates must be finite and in [0,1]")
        key = (entry["edit"], seed, target, condition)
        if key in seen:
            raise ValueError(f"Duplicate D3 pair {key}")
        seen.add(key)
        group_key = (entry["edit"], seed)
        group = groups.setdefault(group_key, {
            "prompt_id": prompt["id"], "prompt_index": prompt["index"],
            "base_prompt": prompt["base"], "edit_prompt": prompt["edit"],
            "probe": prompt.get("probe"), "evidence": prompt.get("evidence", "directional"),
            "seed": seed, "targets": [], "conditions_by_target": {}})
        group["conditions_by_target"].setdefault(target, []).append(condition)
    if not groups:
        raise ValueError("manifest has no cases")
    for group in groups.values():
        group["targets"] = sorted(group["conditions_by_target"])
    return list(groups.values())


def validate_screen(groups, plan):
    manifest = plan["manifest"]
    expected_units = {(edit, seed) for edit in manifest["edits"] for seed in manifest["seeds"]}
    if {(group["prompt_id"], group["seed"]) for group in groups} != expected_units:
        raise ValueError("D3 edit/seed grid differs from the frozen plan")
    for group in groups:
        if set(group["targets"]) != set(manifest["chunks"]):
            raise ValueError("D3 chunk grid differs from the frozen plan")
        if any(set(values) != set(manifest["conditions"])
               for values in group["conditions_by_target"].values()):
            raise ValueError("D3 condition grid differs from the frozen plan")
    count = sum(len(values) for group in groups for values in group["conditions_by_target"].values())
    if count != manifest["expected_pairs"]:
        raise ValueError("D3 pair count differs from the frozen plan")
