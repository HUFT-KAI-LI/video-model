"""Fail-closed held-out analysis for M1-A mask distillation."""
import statistics

def _dominates(point, baseline, tolerance=1e-12):
    return (point["delta_R"] >= baseline["delta_R"] - tolerance and
            point["D_drift"] <= baseline["D_drift"] + tolerance and
            (point["delta_R"] > baseline["delta_R"] + tolerance or
             point["D_drift"] < baseline["D_drift"] - tolerance))


def analyze(records, plan, sources):
    final_conditions = tuple(plan["heldout"]["conditions"])
    expected = {(edit, seed, 4) for edit in plan["teacher"]["edits"]
                for seed in plan["heldout"]["seeds"]}
    grouped = {}
    for record in records:
        if record.get("protocol") != plan["protocol"]:
            raise ValueError("mask-distillation protocol mismatch")
        key = (record["prompt_id"], record["seed"], record["target_chunk"])
        if record["condition"] in grouped.setdefault(key, {}):
            raise ValueError(f"duplicate held-out condition {key}")
        grouped[key][record["condition"]] = record
    if set(grouped) != expected:
        raise ValueError("held-out unit grid incomplete")
    units, prompt_only_masks = [], {}
    wins = {"prompt_only": 0, "prompt_state": 0}
    gamma, drift_lambda = plan["teacher"]["gamma_l1"], plan["teacher"]["oracle_lambda"]
    for key, cases in sorted(grouped.items()):
        if set(cases) != set(final_conditions) or len({c["checkpoint_sha256"] for c in cases.values()}) != 1:
            raise ValueError(f"{key}: incomplete conditions or checkpoint mismatch")
        for condition, case in cases.items():
            mask = case["layer_release"]
            if len(mask) != 30 or any(not 0 <= value <= 1 for value in mask):
                raise ValueError(f"{key}: invalid mask for {condition}")
            partition = case.get("history_partition", {})
            if set(partition) != {"replay", "text_rebind"} or any(
                    value["latent_frames"] != {"history": 9, "current": 3}
                    or value["modules_checked"] != 30 for value in partition.values()):
                raise ValueError(f"{key}: routing audit failed for {condition}")
            if any(not value.get("exact") for value in case["rng"].values()):
                raise ValueError(f"{key}: RNG mismatch for {condition}")
            if not case["preservation"]["outside_exact"] or case["preservation"]["identity"]["status"] != "measured":
                raise ValueError(f"{key}: preservation diagnostic missing for {condition}")
            effect = case["responsiveness"]["text_rebind"]["S_proxy"] - case["responsiveness"]["replay"]["S_proxy"]
            if abs(effect - case["editability"]["E"]) > 1e-12:
                raise ValueError(f"{key}: stored E mismatch for {condition}")
        if not cases["full"]["drift"]["exact"] or cases["full"]["D_drift"] != 0:
            raise ValueError(f"{key}: full P0 is not exact")
        if (cases["full"]["layer_release"] != [0.] * 30
                or cases["global_.5"]["layer_release"] != [.5] * 30
                or ("current_only" in cases and cases["current_only"]["layer_release"] != [1.] * 30)):
            raise ValueError(f"{key}: fixed baseline masks changed")
        previous = prompt_only_masks.setdefault(key[0], cases["prompt_only"]["layer_release"])
        if previous != cases["prompt_only"]["layer_release"]:
            raise ValueError(f"{key[0]}: prompt-only mask depends on held-out state")
        full_r = cases["full"]["editability"]["R"]
        if full_r is None:
            raise ValueError(f"{key}: invalid full-reference effect")
        points = {}
        for condition, case in cases.items():
            loss = (-case["editability"]["E"] + drift_lambda * case["D_drift"]
                    + gamma * sum(case["layer_release"]))
            points[condition] = {"R": case["editability"]["R"],
                                 "delta_R": case["editability"]["R"] - full_r,
                                 "D_drift": case["D_drift"], "oracle_loss": loss}
        unit_wins = {}
        for controller in ("prompt_only", "prompt_state"):
            unit_wins[controller] = _dominates(points[controller], points["global_.5"])
            wins[controller] += int(unit_wins[controller])
        denominator = points["global_.5"]["oracle_loss"] - points["oracle"]["oracle_loss"]
        recovery = {controller: ((points["global_.5"]["oracle_loss"] - points[controller]["oracle_loss"]) / denominator
                                 if denominator > 0 else None)
                    for controller in ("prompt_only", "prompt_state")}
        units.append({"prompt_id": key[0], "seed": key[1], "points": points,
                      "pareto_wins": unit_wins, "oracle_advantage_recovery": recovery})
    aggregate = {}
    for condition in final_conditions:
        values = [unit["points"][condition] for unit in units]
        aggregate[condition] = {"median_delta_R": statistics.median(v["delta_R"] for v in values),
                                "median_D_drift": statistics.median(v["D_drift"] for v in values)}
    threshold = int(plan["heldout"].get("required_pareto_wins", 6))
    passing = [name for name, count in wins.items() if count >= threshold]
    primary = plan["heldout"].get("primary_controller")
    decision = "PASS" if ((primary in passing) if primary else bool(passing)) else "NO_PASS"
    recoveries = {name: [unit["oracle_advantage_recovery"][name] for unit in units
                         if unit["oracle_advantage_recovery"][name] is not None]
                  for name in wins}
    return {"protocol": plan["protocol"], "decision": decision,
            "primary_controller": primary,
            "passing_controllers": passing, "pareto_wins": wins, "required_wins": threshold,
            "aggregate_descriptive": aggregate,
            "median_oracle_advantage_recovery": {
                name: statistics.median(values) if values else None for name, values in recoveries.items()},
            "unit_results": units, "sources": sources,
            "mask_mse_role": "training_diagnostic_only"}
