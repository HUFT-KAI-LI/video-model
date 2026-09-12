"""Fail-closed descriptive analysis for the exploratory D3 path screen."""
from __future__ import annotations

import math
import statistics

from .history_component_analysis import _valid_dino_distance
from .history_paths import CONDITIONS, PROTOCOL, condition_spec


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
        raise ValueError("D3 plan pair count disagrees with its grid")
    indexed = {}
    for record in records:
        key = _key(record)
        if key in indexed:
            raise ValueError(f"duplicate D3 case {key}")
        indexed[key] = record
    if set(indexed) != expected:
        raise ValueError(f"incomplete D3 grid: missing={sorted(expected-set(indexed))} "
                         f"extra={sorted(set(indexed)-expected)}")
    for key, record in indexed.items():
        condition = key[3]
        if record.get("protocol") != PROTOCOL:
            raise ValueError(f"protocol mismatch for {key}")
        spec = condition_spec(condition)
        if record.get("mechanism") == "stage_c_global":
            observed = {"kind": "global", "gate": record.get("history_gate")}
        elif record.get("mechanism") == "attention_path":
            gates = record.get("history_path_gates", {})
            observed = {"kind": "path", "score_gate": gates.get("score"),
                        "value_gate": gates.get("value")}
        else:
            raise ValueError(f"unknown mechanism for {key}")
        if observed != spec:
            raise ValueError(f"condition mechanism mismatch for {key}")
        if record.get("evidence") != "directional":
            raise ValueError(f"D3 contains non-directional edit {key[0]}")
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
        if condition == "full" and (not record["sanity"].get("full_history_P0_exact_base")
                                    or abs(drift) > tolerance):
            raise ValueError(f"full path baseline is not exact for {key}")
    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            cases = {name: indexed[(edit, seed, 4, name)] for name in CONDITIONS}
            if len({case["cache"]["sha256"] for case in cases.values()}) != 1:
                raise ValueError(f"conditions do not share one checkpoint for {(edit, seed)}")
            full_scores = [case["editability"]["S_full"] for case in cases.values()]
            if max(full_scores) - min(full_scores) > tolerance:
                raise ValueError(f"S_full is condition-dependent for {(edit, seed)}")
            full_hashes = cases["full"]["chunk_latent_sha256"]
            for name, case in cases.items():
                if condition_spec(name)["kind"] == "global":
                    if case.get("history_partition") is not None:
                        raise ValueError(f"global condition used path routing for {(edit, seed, name)}")
                    continue
                audits = case.get("history_partition", {})
                if set(audits) != {"replay", "text_rebind"} or any(
                        audits[p].get("latent_frames") != {"history": 9, "current": 3}
                        for p in ("replay", "text_rebind")):
                    raise ValueError(f"path partition audit failed for {(edit, seed, name)}")
            if all(cases["current_only"]["chunk_latent_sha256"][p] == full_hashes[p]
                   for p in ("replay", "text_rebind")):
                raise ValueError(f"current-only is inert for {(edit, seed)}")
            for name in ("score_.5", "value_.5", "value_0", "score_.5_value_.5"):
                if all(cases[name]["chunk_latent_sha256"][p] == full_hashes[p]
                       for p in ("replay", "text_rebind")):
                    raise ValueError(f"{name} is inert for {(edit, seed)}")
    return indexed


def analyze(records, plan, provenance):
    indexed = validate_records(records, plan)
    rows, units = [], []
    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            ratios = {name: indexed[(edit, seed, 4, name)]["editability"].get("R")
                      for name in CONDITIONS}
            reference = ratios["full"]
            for name in CONDITIONS:
                record = indexed[(edit, seed, 4, name)]
                ratio = ratios[name]
                rows.append({"edit": edit, "seed": seed, "target_chunk": 4,
                             "condition": name, "E": record["editability"]["E"], "R": ratio,
                             "delta_R": ratio - reference
                             if ratio is not None and reference is not None else None,
                             "D_drift": record["D_drift"]})
            if any(value is None for value in ratios.values()):
                units.append({"edit": edit, "seed": seed,
                              "status": "not_estimable_nonpositive_full_reference"})
                continue
            deltas = {name: value - reference for name, value in ratios.items()}
            units.append({
                "edit": edit, "seed": seed, "status": "estimable", "R": ratios,
                "delta_R": deltas,
                "score_value_inclusion_exclusion_contrast": (
                    ratios["score_.5_value_.5"] - ratios["score_.5"]
                    - ratios["value_.5"] + ratios["full"]),
                "value_zero_competition_gap": ratios["current_only"] - ratios["value_0"],
            })
    estimable = [unit for unit in units if unit["status"] == "estimable"]
    points = {}
    for name in CONDITIONS:
        subset = [row for row in rows if row["condition"] == name]
        points[name] = {
            "median_delta_R": statistics.median(row["delta_R"] for row in subset)
            if all(row["delta_R"] is not None for row in subset) else None,
            "median_D_drift": statistics.median(row["D_drift"] for row in subset)}
    aggregate = {
        "status": "estimable" if len(estimable) == len(units) else "partially_estimable",
        "condition_points": points,
        "median_score_value_inclusion_exclusion_contrast": statistics.median(
            unit["score_value_inclusion_exclusion_contrast"] for unit in estimable)
        if estimable else None,
        "median_value_zero_competition_gap": statistics.median(
            unit["value_zero_competition_gap"] for unit in estimable) if estimable else None,
    }
    preservation = [{
        "edit": record["prompt_id"], "seed": record["seed"],
        "target_chunk": record["target_chunk"], "condition": record["condition"],
        "D_drift": record["D_drift"],
        "dino_appearance": {policy: record["preservation"]["identity"][policy]
                            for policy in ("replay", "text_rebind")},
        "boundary": record["boundary"]}
        for record in records]
    return {"protocol": PROTOCOL, "status": "pass_invariants_exploratory",
            "decision": "descriptive_only_no_automatic_path_classification",
            "provenance": provenance, "pairs": len(records), "rows": rows,
            "path_units": units, "aggregate": aggregate,
            "preservation_rows": preservation,
            "interpretation": {
                "chunk": 4,
                "score_operator": "history_logit_plus_log_alpha_before_joint_softmax",
                "value_operator": "history_values_times_beta_with_logits_unchanged",
                "claim_limit": "exploratory path localization; no significance test."}}
