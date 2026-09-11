"""Frozen paired and Pareto analysis for the history-release sweep."""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict


def exact_positive_sign_p(positive: int, negative: int) -> float:
    """One-sided exact P(X >= positive), ties excluded."""
    n = positive + negative
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(positive, n + 1)) / (2 ** n)


def percentile_bootstrap_median(values, samples: int, seed: int, confidence: float):
    rng = random.Random(seed)
    n = len(values)
    draws = []
    for _ in range(samples):
        draws.append(statistics.median(values[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    tail = (1 - confidence) / 2
    low = draws[max(0, int(tail * samples))]
    high = draws[min(samples - 1, int((1 - tail) * samples) - 1)]
    return {"median": statistics.median(values), "low": low, "high": high,
            "confidence": confidence, "samples": samples, "n": n}


def _mean(values):
    values = list(values)
    return sum(values) / len(values)


def _case_key(record):
    return (record["prompt_id"], int(record["seed"]),
            int(record["target_chunk"]), float(record["history_gate"]))


def expected_keys(plan):
    manifest = plan["manifest"]
    return {(edit, seed, chunk, float(gate))
            for edit in manifest["edits"] for seed in manifest["seeds"]
            for chunk in manifest["calibration_chunks"] + manifest["confirmatory_chunks"]
            for gate in manifest["gates"]}


def validate_records(records, plan):
    expected = expected_keys(plan)
    if len(expected) != plan["manifest"]["expected_pairs"]:
        raise ValueError("analysis plan expected_pairs disagrees with its case grid")
    indexed = {}
    for record in records:
        key = _case_key(record)
        if key in indexed:
            raise ValueError(f"duplicate sweep case {key}")
        indexed[key] = record
    missing, extra = expected - indexed.keys(), indexed.keys() - expected
    if missing or extra:
        raise ValueError(f"incomplete case grid: missing={sorted(missing)} extra={sorted(extra)}")

    tolerance = float(plan["confirmatory"]["zero_tolerance"])
    for key, record in indexed.items():
        edit, seed, chunk, gate = key
        if record.get("protocol") != plan["protocol"]:
            raise ValueError(f"protocol mismatch for {key}")
        if record.get("reference_history_gate") != 1.0:
            raise ValueError(f"reference generation was not fixed at g=1 for {key}")
        if record.get("evidence") != "directional":
            raise ValueError(f"confirmatory manifest contains non-directional edit {edit}")
        if not record["preservation"].get("outside_exact"):
            raise ValueError(f"outside preservation failed for {key}")
        if not all(record["rng"][policy].get("exact") for policy in ("replay", "text_rebind")):
            raise ValueError(f"replay noise mismatch for {key}")
        response = record["responsiveness"]
        calculated_e = response["text_rebind"]["S_proxy"] - response["replay"]["S_proxy"]
        if abs(calculated_e - record["editability"]["E"]) > tolerance:
            raise ValueError(f"stored E mismatch for {key}")
        full = record["editability"]["S_full"]
        ratio = record["editability"].get("R_k")
        expected_ratio = calculated_e / full if full > 0 else None
        if ((expected_ratio is None) != (ratio is None) or
                expected_ratio is not None and abs(expected_ratio - ratio) > tolerance):
            raise ValueError(f"stored R mismatch for {key}")
        drift = float(record["D_drift"])
        if not math.isfinite(drift) or drift < 0:
            raise ValueError(f"invalid D_drift for {key}")
        if gate == 1.0:
            sanity = record["sanity"]
            if not sanity.get("g1_P0_exact_base"):
                raise ValueError(f"g=1 P0 exact-base invariant failed for {key}")
            if sanity.get("g1_P1_exact_sealed") is False:
                raise ValueError(f"available historical g=1 seal failed for {key}")
            if abs(drift) > tolerance:
                raise ValueError(f"D_drift(1) is nonzero for {key}")

    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            for chunk in plan["manifest"]["calibration_chunks"]:
                calibration = [indexed[(edit, seed, chunk, float(g))]
                               for g in plan["manifest"]["gates"]]
                for policy in ("replay", "text_rebind"):
                    hashes = {case["chunk_latent_sha256"][policy] for case in calibration}
                    if len(hashes) != 1:
                        raise ValueError(f"chunk-0 {policy} is gate-dependent for {(edit, seed)}")
            for chunk in plan["manifest"]["confirmatory_chunks"]:
                cases = [indexed[(edit, seed, chunk, float(g))]
                         for g in plan["manifest"]["gates"]]
                full_values = [case["editability"]["S_full"] for case in cases]
                if max(full_values) - min(full_values) > tolerance:
                    raise ValueError(f"S_full is gate-dependent for {(edit, seed, chunk)}")
                if all(len({case["chunk_latent_sha256"][policy] for case in cases}) == 1
                       for policy in ("replay", "text_rebind")):
                    raise ValueError(
                        f"history gate intervention is inert for {(edit, seed, chunk)}")
    return indexed


def paired_rows(indexed, plan):
    rows = []
    chunks = plan["manifest"]["confirmatory_chunks"]
    for edit in plan["manifest"]["edits"]:
        for seed in plan["manifest"]["seeds"]:
            for chunk in chunks:
                reference_scores = indexed[(edit, seed, chunk, 1.0)]["editability"]
                reference = reference_scores["E"]
                reference_r = reference_scores.get("R_k")
                for gate in plan["manifest"]["gates"]:
                    record = indexed[(edit, seed, chunk, float(gate))]
                    score = record["editability"]
                    ratio = score.get("R_k")
                    rows.append({"edit": edit, "seed": seed, "target_chunk": chunk,
                                 "history_gate": float(gate),
                                 "E": score["E"],
                                 "E_reference_g1": reference,
                                 "delta_E": score["E"] - reference,
                                 "R": ratio, "R_reference_g1": reference_r,
                                 "delta_R": ratio - reference_r
                                 if ratio is not None and reference_r is not None else None,
                                 "D_drift": record["D_drift"]})
    return rows


def cluster_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["edit"], row["seed"], row["history_gate"])].append(row)
    return [{"edit": edit, "seed": seed, "history_gate": gate,
             "delta_E": _mean(row["delta_E"] for row in values),
             "delta_R": _mean(row["delta_R"] for row in values)
             if all(row["delta_R"] is not None for row in values) else None,
             "D_drift": _mean(row["D_drift"] for row in values),
             "chunks": sorted(row["target_chunk"] for row in values)}
            for (edit, seed, gate), values in sorted(grouped.items())]


def pareto_report(clusters, plan):
    gates = [float(g) for g in plan["manifest"]["gates"]]
    tolerance = float(plan["pareto"]["dominance_tolerance"])
    points = {}
    by_gate = {gate: [row for row in clusters if row["history_gate"] == gate] for gate in gates}
    invalid = [row for row in clusters if row["delta_R"] is None]
    if invalid:
        return {"status": "not_estimable_nonpositive_full_reference",
                "invalid_units": [{key: row[key] for key in ("edit", "seed", "history_gate")}
                                  for row in invalid],
                "points": [], "frontier_gates": [], "paired_dominance": [],
                "operating_point": None, "decision_bearing": False}
    for gate, rows in by_gate.items():
        points[gate] = {"history_gate": gate,
                        "normalized_editability": statistics.median(row["delta_R"] for row in rows),
                        "drift": statistics.median(row["D_drift"] for row in rows)}

    def dominates(a, b):
        no_worse = (a["normalized_editability"] >= b["normalized_editability"] - tolerance and
                    a["drift"] <= b["drift"] + tolerance)
        strict = (a["normalized_editability"] > b["normalized_editability"] + tolerance or
                  a["drift"] < b["drift"] - tolerance)
        return no_worse and strict

    frontier = [gate for gate in gates
                if not any(dominates(points[other], points[gate])
                           for other in gates if other != gate)]
    paired = []
    minimum = int(plan["pareto"]["paired_dominance_minimum_clusters"])
    for gate_a in gates:
        keyed_a = {(row["edit"], row["seed"]): row for row in by_gate[gate_a]}
        for gate_b in gates:
            if gate_a == gate_b:
                continue
            count = 0
            for unit, a in keyed_a.items():
                b = next(row for row in by_gate[gate_b]
                         if (row["edit"], row["seed"]) == unit)
                no_worse = (a["delta_R"] >= b["delta_R"] - tolerance and
                            a["D_drift"] <= b["D_drift"] + tolerance)
                strict = (a["delta_R"] > b["delta_R"] + tolerance or
                          a["D_drift"] < b["D_drift"] - tolerance)
                count += bool(no_worse and strict)
            paired.append({"gate_a": gate_a, "gate_b": gate_b,
                           "dominance_clusters": count,
                           "paired_dominates": count >= minimum})
    return {"status": "complete", "editability_coordinate": "median_cluster_delta_R",
            "points": [points[gate] for gate in gates], "frontier_gates": frontier,
            "paired_dominance": paired, "operating_point": None,
            "decision_bearing": False}


def analyze(records, plan):
    indexed = validate_records(records, plan)
    rows = paired_rows(indexed, plan)
    clusters = cluster_rows(rows)
    primary_gate = float(plan["confirmatory"]["gate"])
    primary = [row for row in clusters if row["history_gate"] == primary_gate]
    tolerance = float(plan["confirmatory"]["zero_tolerance"])
    positive = sum(row["delta_E"] > tolerance for row in primary)
    negative = sum(row["delta_E"] < -tolerance for row in primary)
    ties = len(primary) - positive - negative
    p_value = exact_positive_sign_p(positive, negative)
    median_effect = statistics.median(row["delta_E"] for row in primary)

    replication = {"chunks": [], "edits": []}
    required_chunk = int(plan["confirmatory"]["replication"]["minimum_positive_per_chunk"])
    for chunk in plan["manifest"]["confirmatory_chunks"]:
        values = [row["delta_E"] for row in rows
                  if row["history_gate"] == primary_gate and row["target_chunk"] == chunk]
        replication["chunks"].append({"target_chunk": chunk, "median_delta_E": statistics.median(values),
                                      "positive": sum(value > tolerance for value in values),
                                      "passed": statistics.median(values) > tolerance and
                                                sum(value > tolerance for value in values) >= required_chunk})
    for edit in plan["manifest"]["edits"]:
        values = [row["delta_E"] for row in primary if row["edit"] == edit]
        effect = _mean(values)
        replication["edits"].append({"edit": edit, "mean_cluster_delta_E": effect,
                                     "passed": effect > tolerance})
    positive_edits = sum(row["passed"] for row in replication["edits"])
    criteria = {"complete_invariants": True,
                "positive_clusters": positive >= plan["confirmatory"]["minimum_positive_clusters"],
                "sign_test": p_value <= plan["confirmatory"]["alpha"],
                "positive_median": median_effect > tolerance,
                "chunk_replication": all(row["passed"] for row in replication["chunks"]),
                "edit_replication": positive_edits >=
                                    plan["confirmatory"]["replication"]["minimum_positive_edits"]}
    uncertainty = plan["uncertainty"]
    return {"schema": 1, "status": "complete", "estimand": "delta_E=E(g)-E(1)",
            "confirmatory": {"gate": primary_gate, "passed": all(criteria.values()),
                              "criteria": criteria, "positive": positive,
                              "negative": negative, "ties": ties,
                              "exact_one_sided_sign_p": p_value,
                              "median_cluster_delta_E": median_effect,
                              "bootstrap": percentile_bootstrap_median(
                                  [row["delta_E"] for row in primary],
                                  int(uncertainty["bootstrap_samples"]),
                                  int(uncertainty["bootstrap_seed"]),
                                  float(uncertainty["confidence"])),
                              "replication": replication},
            "historical_sealed_g1": {
                "available": sum(indexed[key]["sanity"].get("g1_P1_exact_sealed") is True
                                 for key in indexed if key[3] == 1.0),
                "unavailable": sum(indexed[key]["sanity"].get("g1_P1_exact_sealed") is None
                                   for key in indexed if key[3] == 1.0),
                "failed": 0},
            "pareto": pareto_report(clusters, plan),
            "paired_rows": rows, "cluster_rows": clusters,
            "claim": plan["claim_scope"]["confirmatory_pass"] if all(criteria.values())
                     else plan["claim_scope"]["confirmatory_fail"]}
