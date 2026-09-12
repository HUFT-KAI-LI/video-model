"""Fail-closed analysis for the M0/M1 oracle layer-mask experiment."""
from __future__ import annotations

import statistics

from .oracle_layer_mask import BASELINES, oracle_name


def _dominates(point, baseline, tolerance=1e-12):
    gain_ok = point["delta_R"] >= baseline["delta_R"] - tolerance
    drift_ok = point["D_drift"] <= baseline["D_drift"] + tolerance
    strict = (point["delta_R"] > baseline["delta_R"] + tolerance or
              point["D_drift"] < baseline["D_drift"] - tolerance)
    return bool(gain_ok and drift_ok and strict)


def analyze(records, plan, sources):
    lambdas = [float(value) for value in plan["optimization"]["lambdas"]]
    conditions = set(BASELINES) | {oracle_name(value) for value in lambdas}
    expected_units = {(edit, seed, 4) for edit in plan["manifest"]["edits"]
                      for seed in plan["manifest"]["seeds"]}
    grouped = {}
    for record in records:
        if record.get("protocol") != plan["protocol"]:
            raise ValueError("protocol mismatch")
        key = (record["prompt_id"], record["seed"], record["target_chunk"])
        if record["condition"] in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate oracle condition {key + (record['condition'],)}")
        grouped[key][record["condition"]] = record
    if set(grouped) != expected_units:
        raise ValueError("oracle unit grid is incomplete")
    units = []
    for key, cases in sorted(grouped.items()):
        if set(cases) != conditions:
            raise ValueError(f"{key}: final condition grid is incomplete")
        checkpoints = {case["checkpoint_sha256"] for case in cases.values()}
        if len(checkpoints) != 1:
            raise ValueError(f"{key}: candidates do not share one checkpoint")
        for condition, case in cases.items():
            mask = case["layer_release"]
            if len(mask) != 30 or any(not 0 <= value <= 1 for value in mask):
                raise ValueError(f"{key}: invalid 30-layer mask for {condition}")
            partitions = case.get("history_partition", {})
            if set(partitions) != {"replay", "text_rebind"} or any(
                    value["latent_frames"] != {"history": 9, "current": 3}
                    or value["modules_checked"] != 30 for value in partitions.values()):
                raise ValueError(f"{key}: layer routing audit failed for {condition}")
            if not case.get("preservation", {}).get("outside_exact"):
                raise ValueError(f"{key}: outside preservation failed for {condition}")
            if case.get("preservation", {}).get("identity", {}).get("status") != "measured":
                raise ValueError(f"{key}: DINO missing for {condition}")
            if any(not value.get("exact") for value in case.get("rng", {}).values()):
                raise ValueError(f"{key}: RNG mismatch for {condition}")
            effect = (case["responsiveness"]["text_rebind"]["S_proxy"] -
                      case["responsiveness"]["replay"]["S_proxy"])
            if abs(effect - case["editability"]["E"]) > 1e-12:
                raise ValueError(f"{key}: stored E mismatch for {condition}")
        if not cases["full"]["drift"]["exact"] or cases["full"]["D_drift"] != 0:
            raise ValueError(f"{key}: full P0 is not exact base")
        if cases["global_.5"]["layer_release"] != [0.5] * 30:
            raise ValueError(f"{key}: global baseline is not all-half")
        if cases["full"]["layer_release"] != [0.0] * 30:
            raise ValueError(f"{key}: full baseline is not all-zero")
        if cases["current_only"]["layer_release"] != [1.0] * 30:
            raise ValueError(f"{key}: current baseline is not all-one")
        if all(cases["current_only"]["chunk_latent_sha256"][name] ==
               cases["full"]["chunk_latent_sha256"][name]
               for name in ("replay", "text_rebind")):
            raise ValueError(f"{key}: current-only intervention is inert")
        full_r = cases["full"]["editability"]["R"]
        if full_r is None:
            raise ValueError(f"{key}: nonpositive full reference")
        points = {condition: {"R": case["editability"]["R"],
                              "delta_R": case["editability"]["R"] - full_r,
                              "D_drift": case["D_drift"]}
                  for condition, case in cases.items()}
        baseline = points["global_.5"]
        dominated_by = [name for name in conditions if name.startswith("oracle_")
                        and _dominates(points[name], baseline)]
        units.append({"prompt_id": key[0], "seed": key[1], "target_chunk": key[2],
                      "points": points, "oracle_dominates_global": bool(dominated_by),
                      "dominating_conditions": sorted(dominated_by)})
    by_condition = {}
    for condition in sorted(conditions):
        points = [unit["points"][condition] for unit in units]
        by_condition[condition] = {
            "median_delta_R": statistics.median(point["delta_R"] for point in points),
            "median_D_drift": statistics.median(point["D_drift"] for point in points)}
    wins = sum(unit["oracle_dominates_global"] for unit in units)
    return {"protocol": plan["protocol"], "decision": "PASS" if wins else "NO_PASS",
            "pass_rule": plan["estimands"]["pass"], "units_with_oracle_pareto_gain": wins,
            "total_units": len(units), "unit_results": units,
            "aggregate_descriptive": by_condition, "sources": sources,
            "interpretation": "exploratory_oracle_feasibility_no_significance_test"}
