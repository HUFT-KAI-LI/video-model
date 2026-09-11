"""Validated D1 sink/old/recent history-component interventions."""
import math

from . import edit_experiment as ex


PROTOCOL = "fixed_history_components_d1_v1"
COMPONENTS = ("sink", "old", "recent")
CONDITIONS = {
    "full_history": {"kind": "components", "gates": {"sink": 1.0, "old": 1.0, "recent": 1.0}},
    "global_release": {"kind": "global", "gate": 0.5},
    "sink_release": {"kind": "components", "gates": {"sink": 0.5, "old": 1.0, "recent": 1.0}},
    "old_release": {"kind": "components", "gates": {"sink": 1.0, "old": 0.5, "recent": 1.0}},
    "recent_release": {"kind": "components", "gates": {"sink": 1.0, "old": 1.0, "recent": 0.5}},
    "non_sink_release": {"kind": "components", "gates": {"sink": 1.0, "old": 0.5, "recent": 0.5}},
}


def condition_spec(name):
    if name not in CONDITIONS:
        raise ValueError(f"Unknown history-component condition {name}")
    spec = CONDITIONS[name]
    if spec["kind"] == "global":
        return {"kind": "global", "gate": float(spec["gate"])}
    return {"kind": "components",
            "gates": {component: float(spec["gates"][component]) for component in COMPONENTS}}


def manifest_groups(config, manifest):
    if manifest.get("schema") != 1 or manifest.get("experiment") != "history_component_screen_d1":
        raise ValueError("Use the D1 history-component manifest schema")
    entries = manifest.get("cases", [])
    if not entries:
        raise ValueError("manifest has no cases")
    prompts = {prompt["id"]: prompt for prompt in ex.prompt_cases(config)}
    groups, seen = {}, set()
    for entry in entries:
        if set(entry) != {"edit", "seed", "target_chunk", "condition"}:
            raise ValueError("D1 cases must contain only edit, seed, target_chunk, and condition")
        prompt = prompts.get(entry["edit"])
        if prompt is None:
            raise ValueError(f"Unknown edit {entry['edit']}")
        seed, target = entry["seed"], entry["target_chunk"]
        if type(seed) is not int or type(target) is not int or target < 1:
            raise ValueError("seed must be an integer and target_chunk must be a positive integer")
        condition = entry["condition"]
        spec = condition_spec(condition)
        values = spec.get("gates", {"global": spec.get("gate")}).values()
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
            raise ValueError("history gates must be finite and in [0,1]")
        key = (entry["edit"], seed, target, condition)
        if key in seen:
            raise ValueError(f"Duplicate D1 pair {key}")
        seen.add(key)
        group_key = (entry["edit"], seed)
        if group_key not in groups:
            groups[group_key] = {
                "prompt_id": prompt["id"], "prompt_index": prompt["index"],
                "base_prompt": prompt["base"], "edit_prompt": prompt["edit"],
                "probe": prompt.get("probe"),
                "evidence": prompt.get("evidence", "directional"),
                "seed": seed, "targets": [], "conditions_by_target": {},
            }
        groups[group_key]["conditions_by_target"].setdefault(target, []).append(condition)
    for group in groups.values():
        group["targets"] = sorted(group["conditions_by_target"])
    return list(groups.values())


def validate_screen(groups, plan):
    expected_edits = set(plan["manifest"]["edits"])
    expected_seeds = set(plan["manifest"]["seeds"])
    expected_chunks = set(plan["manifest"]["chunks"])
    expected_conditions = set(plan["manifest"]["conditions"])
    if {(group["prompt_id"], group["seed"]) for group in groups} != {
            (edit, seed) for edit in expected_edits for seed in expected_seeds}:
        raise ValueError("D1 edit/seed grid differs from the frozen screen plan")
    for group in groups:
        if set(group["targets"]) != expected_chunks:
            raise ValueError("D1 chunk grid differs from the frozen screen plan")
        if any(set(values) != expected_conditions
               for values in group["conditions_by_target"].values()):
            raise ValueError("D1 condition grid differs from the frozen screen plan")
    count = sum(len(values) for group in groups
                for values in group["conditions_by_target"].values())
    if count != plan["manifest"]["expected_pairs"]:
        raise ValueError("D1 pair count differs from the frozen screen plan")
