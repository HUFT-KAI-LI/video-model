"""Fail-closed factorial analysis for the exploratory D2 subset screen."""
from __future__ import annotations

import math
import statistics

from .history_component_analysis import _valid_dino_distance
from .history_subsets import CONDITIONS, FACTORIAL_CONDITIONS, PROTOCOL, condition_spec


def _key(record):
    return (record["prompt_id"], int(record["seed"]),
            int(record["target_chunk"]), record["condition"])


def expected_keys(plan):
    grid = plan["manifest"]
    return {(edit, seed, chunk, condition) for edit in grid["edits"]
            for seed in grid["seeds"] for chunk in grid["chunks"]
            for condition in grid["conditions"]}


def validate_records(records, plan, tolerance=1e-12):
    expected = expected_keys(plan)
    if len(expected) != plan["manifest"]["expected_pairs"]:
        raise ValueError("D2 plan pair count disagrees with its grid")
    indexed = {}
    for record in records:
        key = _key(record)
        if key in indexed:
            raise ValueError(f"duplicate D2 case {key}")
        indexed[key] = record
    if expected != set(indexed):
        raise ValueError(f"incomplete D2 grid: missing={sorted(expected-set(indexed))} "
                         f"extra={sorted(set(indexed)-expected)}")
    for key, record in indexed.items():
        condition = key[3]
        if record.get("protocol") != PROTOCOL:
            raise ValueError(f"protocol mismatch for {key}")
        spec = condition_spec(condition)
        observed = ({"kind": "global", "gate": record.get("history_gate")}
                    if record.get("mechanism") == "stage_c_global" else
                    {"kind": "components", "gates": record.get("history_component_gates")})
        if observed != spec:
            raise ValueError(f"condition mechanism mismatch for {key}")
        if record.get("evidence") != "directional":
            raise ValueError(f"D2 contains non-directional edit {key[0]}")
        if not record["sanity"].get("fixed_references_unmodified_full_history"):
            raise ValueError(f"reference invariant failed for {key}")
        if not record["preservation"].get("outside_exact"):
            raise ValueError(f"outside preservation failed for {key}")
        identity = record["preservation"].get("identity", {})
        if identity.get("status") != "measured" or any(
                not _valid_dino_distance(identity.get(policy))
                for policy in ("replay", "text_rebind")):
            raise ValueError(f"invalid DINO diagnostic for {key}")
        if not all(policy in record.get("boundary", {}) for policy in ("replay", "text_rebind")):
            raise ValueError(f"boundary diagnostic missing for {key}")
        if not all(record["rng"][policy].get("exact") for policy in ("replay", "text_rebind")):
            raise ValueError(f"replay noise mismatch for {key}")
        if record.get("cost", {}).get("status") != "invalid_diagnostic":
            raise ValueError(f"timing was treated as valid for {key}")
        response = record["responsiveness"]
        effect = response["text_rebind"]["S_proxy"] - response["replay"]["S_proxy"]
        if abs(effect - record["editability"]["E"]) > tolerance:
            raise ValueError(f"stored E mismatch for {key}")
        full = record["editability"]["S_full"]
        ratio = record["editability"].get("R")
        expected_ratio = effect / full if full > 0 else None
        if ((ratio is None) != (expected_ratio is None) or
                ratio is not None and abs(ratio - expected_ratio) > tolerance):
            raise ValueError(f"stored R mismatch for {key}")
        drift = float(record["D_drift"])
        if not math.isfinite(drift) or drift < 0:
            raise ValueError(f"invalid drift for {key}")
        if condition == "SOR" and (not record["sanity"].get("full_history_P0_exact_base")
                                   or abs(drift) > tolerance):
            raise ValueError(f"SOR baseline is not exact for {key}")
    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            cases = {name: indexed[(edit, seed, 4, name)] for name in CONDITIONS}
            if len({case["cache"]["sha256"] for case in cases.values()}) != 1:
                raise ValueError(f"conditions do not share one checkpoint for {(edit, seed)}")
            if max(case["editability"]["S_full"] for case in cases.values()) - min(
                    case["editability"]["S_full"] for case in cases.values()) > tolerance:
                raise ValueError(f"S_full is condition-dependent for {(edit, seed)}")
            expected_partition = {"sink": 3, "old": 3, "recent": 3, "current": 3}
            for name, case in cases.items():
                if name == "global_release":
                    if case.get("history_partition") is not None:
                        raise ValueError("global reference used component routing")
                    continue
                partitions = case.get("history_partition", {})
                if set(partitions) != {"replay", "text_rebind"} or any(
                        partitions[p].get("latent_frames") != expected_partition
                        for p in ("replay", "text_rebind")):
                    raise ValueError(f"partition audit failed for {(edit, seed, name)}")
            if all(cases["empty"]["chunk_latent_sha256"][p] ==
                   cases["SOR"]["chunk_latent_sha256"][p]
                   for p in ("replay", "text_rebind")):
                raise ValueError(f"current-only endpoint is inert for {(edit, seed)}")
    return indexed


def _interaction(values):
    return {
        "I_SO": values["SO"] - values["S"] - values["O"] + values["empty"],
        "I_SR": values["SR"] - values["S"] - values["R"] + values["empty"],
        "I_OR": values["OR"] - values["O"] - values["R"] + values["empty"],
        "I_SOR": (values["SOR"] - values["SO"] - values["SR"] - values["OR"]
                  + values["S"] + values["O"] + values["R"] - values["empty"]),
    }


def _shapley(values):
    locking = {name: values["empty"] - value for name, value in values.items()}
    phi_s = (locking["S"] / 3 + (locking["SO"] - locking["O"]
             + locking["SR"] - locking["R"]) / 6
             + (locking["SOR"] - locking["OR"]) / 3)
    phi_o = (locking["O"] / 3 + (locking["SO"] - locking["S"]
             + locking["OR"] - locking["R"]) / 6
             + (locking["SOR"] - locking["SR"]) / 3)
    phi_r = (locking["R"] / 3 + (locking["SR"] - locking["S"]
             + locking["OR"] - locking["O"]) / 6
             + (locking["SOR"] - locking["SO"]) / 3)
    return locking, {"sink": phi_s, "old": phi_o, "recent": phi_r}


def analyze(records, plan, provenance):
    indexed = validate_records(records, plan)
    rows, units = [], []
    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            scores = {}
            for condition in CONDITIONS:
                record = indexed[(edit, seed, 4, condition)]
                score = record["editability"]
                scores[condition] = score.get("R")
                rows.append({"edit": edit, "seed": seed, "target_chunk": 4,
                             "condition": condition, "E": score["E"], "R": score.get("R"),
                             "delta_R": score.get("R") - indexed[(edit, seed, 4, "SOR")]["editability"].get("R")
                             if score.get("R") is not None and indexed[(edit, seed, 4, "SOR")]["editability"].get("R") is not None else None,
                             "D_drift": record["D_drift"]})
            factorial = {name: scores[name] for name in FACTORIAL_CONDITIONS}
            if any(value is None for value in factorial.values()):
                units.append({"edit": edit, "seed": seed, "status": "not_estimable_nonpositive_full_reference"})
                continue
            interactions = _interaction(factorial)
            locking, shapley = _shapley(factorial)
            residual = sum(shapley.values()) - locking["SOR"]
            if abs(residual) > 1e-10:
                raise ValueError(f"Shapley efficiency failed for {(edit, seed)}")
            units.append({"edit": edit, "seed": seed, "status": "estimable",
                          "F_R": factorial, "locking": locking,
                          "pairwise_and_third_order_interactions": interactions,
                          "locking_shapley": shapley, "shapley_efficiency_residual": residual})
    estimable = [unit for unit in units if unit["status"] == "estimable"]
    aggregate = {"status": "estimable" if len(estimable) == len(units) else "partially_estimable",
                 "condition_points": {}, "interaction_medians": {}, "locking_shapley_medians": {}}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["condition"] == condition]
        aggregate["condition_points"][condition] = {
            "median_delta_R_vs_SOR": statistics.median(row["delta_R"] for row in subset)
            if all(row["delta_R"] is not None for row in subset) else None,
            "median_D_drift": statistics.median(row["D_drift"] for row in subset)}
    if estimable:
        for name in ("I_SO", "I_SR", "I_OR", "I_SOR"):
            aggregate["interaction_medians"][name] = statistics.median(
                unit["pairwise_and_third_order_interactions"][name] for unit in estimable)
        for name in ("sink", "old", "recent"):
            aggregate["locking_shapley_medians"][name] = statistics.median(
                unit["locking_shapley"][name] for unit in estimable)
    return {"protocol": PROTOCOL, "status": "pass_invariants_exploratory",
            "decision": "descriptive_only_no_automatic_pattern_classification",
            "provenance": provenance, "pairs": len(records), "rows": rows,
            "factorial_units": units, "aggregate": aggregate,
            "interpretation": {
                "chunk": 4, "factorial_operator": "binary_native_attention_subsets",
                "global_release_role": "reference_only_excluded_from_factorial_decomposition",
                "claim_limit": "nonzero contrasts suggest interactions; they do not prove mechanism."}}
