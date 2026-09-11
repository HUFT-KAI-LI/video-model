"""Frozen D2 binary history-subset intervention definitions."""
import math

from . import edit_experiment as ex


PROTOCOL = "fixed_history_subsets_d2_v1"
COMPONENTS = ("sink", "old", "recent")
FACTORIAL_CONDITIONS = ("SOR", "SO", "SR", "OR", "S", "O", "R", "empty")
CONDITIONS = {
    "SOR": {"kind": "components", "gates": {"sink": 1.0, "old": 1.0, "recent": 1.0}},
    "SO": {"kind": "components", "gates": {"sink": 1.0, "old": 1.0, "recent": 0.0}},
    "SR": {"kind": "components", "gates": {"sink": 1.0, "old": 0.0, "recent": 1.0}},
    "OR": {"kind": "components", "gates": {"sink": 0.0, "old": 1.0, "recent": 1.0}},
    "S": {"kind": "components", "gates": {"sink": 1.0, "old": 0.0, "recent": 0.0}},
    "O": {"kind": "components", "gates": {"sink": 0.0, "old": 1.0, "recent": 0.0}},
    "R": {"kind": "components", "gates": {"sink": 0.0, "old": 0.0, "recent": 1.0}},
    "empty": {"kind": "components", "gates": {"sink": 0.0, "old": 0.0, "recent": 0.0}},
    "global_release": {"kind": "global", "gate": 0.5},
}


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError(f"Unknown D2 condition {name}")
    spec = CONDITIONS[name]
    if spec["kind"] == "global":
        return {"kind": "global", "gate": float(spec["gate"])}
    return {"kind": "components",
            "gates": {name: float(spec["gates"][name]) for name in COMPONENTS}}


def manifest_groups(config, manifest):
    if manifest.get("schema") != 1 or manifest.get("experiment") != "history_subset_interaction_d2":
        raise ValueError("Use the D2 history-subset manifest schema")
    prompts = {prompt["id"]: prompt for prompt in ex.prompt_cases(config)}
    groups, seen = {}, set()
    for entry in manifest.get("cases", []):
        if set(entry) != {"edit", "seed", "target_chunk", "condition"}:
            raise ValueError("D2 cases must contain only edit, seed, target_chunk, and condition")
        prompt = prompts.get(entry["edit"])
        if prompt is None:
            raise ValueError(f"Unknown edit {entry['edit']}")
        seed, target, condition = entry["seed"], entry["target_chunk"], entry["condition"]
        if type(seed) is not int or type(target) is not int or target < 1:
            raise ValueError("seed and target_chunk must be positive-grid integers")
        spec = condition_spec(condition)
        values = spec.get("gates", {"global": spec.get("gate")}).values()
        if any(not math.isfinite(value) or value not in (0.0, 1.0)
               for value in values if spec["kind"] == "components"):
            raise ValueError("D2 component gates must be binary")
        key = (entry["edit"], seed, target, condition)
        if key in seen:
            raise ValueError(f"Duplicate D2 pair {key}")
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
        raise ValueError("D2 edit/seed grid differs from the frozen plan")
    for group in groups:
        if set(group["targets"]) != set(manifest["chunks"]):
            raise ValueError("D2 chunk grid differs from the frozen plan")
        if any(set(values) != set(manifest["conditions"])
               for values in group["conditions_by_target"].values()):
            raise ValueError("D2 condition grid differs from the frozen plan")
    count = sum(len(values) for group in groups for values in group["conditions_by_target"].values())
    if count != manifest["expected_pairs"]:
        raise ValueError("D2 pair count differs from the frozen plan")
