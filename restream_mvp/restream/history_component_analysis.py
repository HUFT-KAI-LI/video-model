"""Fail-closed descriptive analysis for the exploratory D1 component screen."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict

from .history_components import CONDITIONS, PROTOCOL, condition_spec


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def _key(record):
    return (record["prompt_id"], int(record["seed"]),
            int(record["target_chunk"]), record["condition"])


def expected_keys(plan):
    manifest = plan["manifest"]
    return {(edit, seed, chunk, condition)
            for edit in manifest["edits"] for seed in manifest["seeds"]
            for chunk in manifest["chunks"] for condition in manifest["conditions"]}


def validate_records(records, plan, tolerance=1e-12):
    expected = expected_keys(plan)
    if len(expected) != plan["manifest"]["expected_pairs"]:
        raise ValueError("D1 plan pair count disagrees with its grid")
    indexed = {}
    for record in records:
        key = _key(record)
        if key in indexed:
            raise ValueError(f"duplicate D1 case {key}")
        indexed[key] = record
    missing, extra = expected - indexed.keys(), indexed.keys() - expected
    if missing or extra:
        raise ValueError(f"incomplete D1 grid: missing={sorted(missing)} extra={sorted(extra)}")

    for key, record in indexed.items():
        edit, seed, chunk, condition = key
        if record.get("protocol") != PROTOCOL:
            raise ValueError(f"protocol mismatch for {key}")
        expected_spec = condition_spec(condition)
        observed = ({"kind": "global", "gate": record.get("history_gate")}
                    if record.get("mechanism") == "stage_c_global" else
                    {"kind": "components", "gates": record.get("history_component_gates")})
        if observed != expected_spec:
            raise ValueError(f"condition mechanism mismatch for {key}")
        if record.get("evidence") != "directional":
            raise ValueError(f"D1 contains non-directional edit {edit}")
        if not record["sanity"].get("fixed_references_unmodified_full_history"):
            raise ValueError(f"reference generation invariant failed for {key}")
        if not record["preservation"].get("outside_exact"):
            raise ValueError(f"outside preservation failed for {key}")
        identity = record["preservation"].get("identity", {})
        if identity.get("status") != "measured":
            raise ValueError(f"DINO appearance diagnostic missing for {key}")
        if any(not math.isfinite(float(identity.get(policy, float("nan"))))
               for policy in ("replay", "text_rebind")):
            raise ValueError(f"invalid DINO appearance diagnostic for {key}")
        if not all(policy in record.get("boundary", {})
                   for policy in ("replay", "text_rebind")):
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
            raise ValueError(f"invalid D_drift for {key}")
        if condition == "full_history":
            if not record["sanity"].get("full_history_P0_exact_base"):
                raise ValueError(f"full-history P0 is not exact for {key}")
            if abs(drift) > tolerance:
                raise ValueError(f"full-history drift is nonzero for {key}")

    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            for chunk in plan["manifest"]["chunks"]:
                cases = {condition: indexed[(edit, seed, chunk, condition)]
                         for condition in plan["manifest"]["conditions"]}
                checkpoints = {case["cache"]["sha256"] for case in cases.values()}
                if len(checkpoints) != 1:
                    raise ValueError(f"conditions do not share one checkpoint for {(edit, seed, chunk)}")
                full_scores = [case["editability"]["S_full"] for case in cases.values()]
                if max(full_scores) - min(full_scores) > tolerance:
                    raise ValueError(f"S_full is condition-dependent for {(edit, seed, chunk)}")
                for policy in ("replay", "text_rebind"):
                    if chunk == 1:
                        if (cases["global_release"]["chunk_latent_sha256"][policy] !=
                                cases["sink_release"]["chunk_latent_sha256"][policy]):
                            raise ValueError(f"chunk1 global != sink for {(edit, seed, policy)}")
                        full_hash = cases["full_history"]["chunk_latent_sha256"][policy]
                        for condition in ("old_release", "recent_release", "non_sink_release"):
                            if cases[condition]["chunk_latent_sha256"][policy] != full_hash:
                                raise ValueError(
                                    f"chunk1 empty {condition} != full for {(edit, seed, policy)}")
                expected_partition = ({"sink": 3, "old": 0, "recent": 0, "current": 3}
                                      if chunk == 1 else
                                      {"sink": 3, "old": 3, "recent": 3, "current": 3})
                for condition, case in cases.items():
                    if condition == "global_release":
                        if case.get("history_partition") is not None:
                            raise ValueError(f"global baseline used component router for {(edit, seed, chunk)}")
                        continue
                    partitions = case.get("history_partition", {})
                    if set(partitions) != {"replay", "text_rebind"}:
                        raise ValueError(f"partition audit missing for {(edit, seed, chunk, condition)}")
                    if any(partitions[policy].get("latent_frames") != expected_partition
                           for policy in ("replay", "text_rebind")):
                        raise ValueError(f"partition audit mismatch for {(edit, seed, chunk, condition)}")
                if chunk == 4:
                    full_hashes = cases["full_history"]["chunk_latent_sha256"]
                    for condition in CONDITIONS:
                        if condition == "full_history":
                            continue
                        if all(cases[condition]["chunk_latent_sha256"][policy] == full_hashes[policy]
                               for policy in ("replay", "text_rebind")):
                            raise ValueError(f"chunk4 {condition} is inert for {(edit, seed)}")
    return indexed


def paired_rows(indexed, plan):
    rows = []
    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            for chunk in plan["manifest"]["chunks"]:
                reference = indexed[(edit, seed, chunk, "full_history")]["editability"]
                for condition in plan["manifest"]["conditions"]:
                    record = indexed[(edit, seed, chunk, condition)]
                    score = record["editability"]
                    ratio, reference_ratio = score.get("R"), reference.get("R")
                    rows.append({
                        "edit": edit, "seed": seed, "target_chunk": chunk,
                        "condition": condition, "E": score["E"],
                        "E_full_history": reference["E"],
                        "delta_E": score["E"] - reference["E"],
                        "R": ratio, "R_full_history": reference_ratio,
                        "delta_R": ratio - reference_ratio
                        if ratio is not None and reference_ratio is not None else None,
                        "D_drift": record["D_drift"],
                    })
    return rows


def cluster_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["edit"], row["seed"], row["condition"])].append(row)
    return [{"edit": edit, "seed": seed, "condition": condition,
             "delta_E": _mean(row["delta_E"] for row in values),
             "delta_R": _mean(row["delta_R"] for row in values)
             if all(row["delta_R"] is not None for row in values) else None,
             "D_drift": _mean(row["D_drift"] for row in values),
             "chunks": sorted(row["target_chunk"] for row in values)}
            for (edit, seed, condition), values in sorted(grouped.items())]


def pareto_report(clusters, plan, tolerance=1e-12):
    conditions = plan["manifest"]["conditions"]
    if any(row["delta_R"] is None for row in clusters):
        return {"status": "not_estimable_nonpositive_full_reference", "points": [],
                "frontier_conditions": [], "selective_dominates_global": []}
    grouped = {condition: [row for row in clusters if row["condition"] == condition]
               for condition in conditions}
    points = {condition: {
        "condition": condition,
        "normalized_editability": statistics.median(row["delta_R"] for row in rows),
        "drift": statistics.median(row["D_drift"] for row in rows),
    } for condition, rows in grouped.items()}

    def dominates(left, right):
        no_worse = (left["normalized_editability"] >= right["normalized_editability"] - tolerance
                    and left["drift"] <= right["drift"] + tolerance)
        strict = (left["normalized_editability"] > right["normalized_editability"] + tolerance
                  or left["drift"] < right["drift"] - tolerance)
        return no_worse and strict

    frontier = [condition for condition in conditions
                if not any(dominates(points[other], points[condition])
                           for other in conditions if other != condition)]
    global_rows = {(row["edit"], row["seed"]): row
                   for row in grouped["global_release"]}
    selective = []
    for condition in plan["screen_interpretation"]["selective_conditions"]:
        support = []
        for row in grouped[condition]:
            reference = global_rows[(row["edit"], row["seed"])]
            support.append({"edit": row["edit"], "seed": row["seed"],
                            "delta_R_selective_minus_global":
                                row["delta_R"] - reference["delta_R"],
                            "drift_selective_minus_global":
                                row["D_drift"] - reference["D_drift"]})
        selective.append({"condition": condition,
                          "aggregate_dominates_global":
                              dominates(points[condition], points["global_release"]),
                          "paired_support": support})
    return {"status": "complete", "points": [points[name] for name in conditions],
            "frontier_conditions": frontier,
            "selective_dominates_global": selective,
            "operating_point": None, "decision_bearing": False}


def analyze(records, plan):
    indexed = validate_records(records, plan)
    rows = paired_rows(indexed, plan)
    clusters = cluster_rows(rows)
    preservation = [{"edit": record["prompt_id"], "seed": record["seed"],
                     "target_chunk": record["target_chunk"],
                     "condition": record["condition"],
                     "D_drift": record["D_drift"],
                     "dino_appearance": {
                         policy: record["preservation"]["identity"][policy]
                         for policy in ("replay", "text_rebind")},
                     "boundary": record["boundary"]}
                    for record in records]
    return {"schema": 1, "status": "complete", "stage": plan["stage"],
            "invariants_passed": True,
            "interpretation": "exploratory_component_screen_no_significance_test",
            "pareto": pareto_report(clusters, plan),
            "paired_rows": rows, "cluster_rows": clusters,
            "preservation_rows": preservation,
            "claim_scope": plan["scope"]["claim"]}
