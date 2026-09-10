"""Aggregate the Edit-Ready MVP runs into ``validation/edit_ready_mvp/summary.json``.

Consumes one or more Experiment A / Experiment B JSON files (possibly sharded
across GPUs) and writes the schema of plan section 16 plus ``timing.json`` and
``cache_manifest.json`` (plan section 15).  Missing inputs are reported as
``null`` and downgrade ``status`` -- nothing is ever pre-filled with fake numbers.

Gate B's automatic aggregate only counts ``evidence: directional`` prompts; the
qualitative prompts (smile / zoom / rain) are reported as a proxy-consistent
change count and stay for human review.  The per-chunk ratio
``R_k = S_text_rebind / S_full_regeneration`` is reported for every target chunk,
including chunk 0, which has no visual history and therefore calibrates the curve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restream import edit_experiment as ex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_many(paths) -> list:
    return [(Path(path), json.loads(Path(path).read_text())) for path in paths]


def mean(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return sum(values) / len(values)


def collect(replay_files, edit_files) -> dict:
    replay_cases, edit_cases, run_records = [], [], []
    for path, data in replay_files + edit_files:
        replay_cases.extend(data.get("cases", [])) if data.get("experiment") == "A_same_prompt_replay" \
            else edit_cases.extend(data.get("cases", []))
        provenance = data.get("provenance") or {}
        run_records.append({"file": path.name, "experiment": data.get("experiment"),
                            "git_commit": provenance.get("git_commit"),
                            "git_dirty": provenance.get("git_dirty"),
                            "code_tree_sha256": provenance.get("code_tree_sha256"),
                            "config_sha256": provenance.get("config_sha256"),
                            "model_hash": provenance.get("model_checkpoint_sha256")})
    return {"replay_cases": replay_cases, "edit_cases": edit_cases, "run_records": run_records}


def boundary_verdict(edit_cases, gate_config) -> dict:
    relative = float(gate_config.get("max_relative_increase", 1.0))
    slack = float(gate_config.get("absolute_slack", 1e-4))
    left, right = [], []
    for case in edit_cases:
        latent = case["boundary"]["latent"]
        left.append((latent["edited"]["left_mse"], latent["base"]["left_mse"]))
        right.append((latent["edited"]["right_mse"], latent["base"]["right_mse"]))

    def summarize(pairs):
        # Drop missing boundaries: chunk 0 has no left neighbour, and non-finite
        # values are stored as JSON null.
        pairs = [(e, b) for e, b in pairs
                 if isinstance(e, (int, float)) and isinstance(b, (int, float))
                 and e == e and b == b]
        if not pairs:
            return {"available": False}
        accepted = [e <= b + max(relative * b, slack) for e, b in pairs]
        base_mean = sum(b for _, b in pairs) / len(pairs)
        edited_mean = sum(e for e, _ in pairs) / len(pairs)
        return {"available": True,
                "delta_mean": sum(e - b for e, b in pairs) / len(pairs),
                "base_mean": base_mean, "edited_mean": edited_mean,
                "relative_increase_of_means": ((edited_mean - base_mean) / base_mean)
                if base_mean > 0 else None,
                "acceptable": all(accepted), "acceptable_cases": sum(accepted),
                "total_cases": len(pairs),
                "worst_relative_increase": max((e - b) / b for e, b in pairs if b > 0)
                if any(b > 0 for _, b in pairs) else None}

    return {"left": summarize(left), "right": summarize(right),
            "max_relative_increase": relative, "absolute_slack": slack}


def _side_ok(side, fraction):
    if not side.get("available"):
        return True
    total = side.get("total_cases") or 0
    if not total:
        return bool(side.get("acceptable"))
    return (side.get("acceptable_cases", 0) / total) >= fraction


def decision(replay_passed, edit_passed, strong_cases, successful_cases, boundary, cost,
             acceptable_fraction=0.9, min_strong_fraction=0.25) -> str:
    """Verdict following the plan's section 13 interpretation matrix."""
    if not replay_passed:
        return "NO_GO_STATE_INCOMPLETE"
    if not edit_passed:
        return "NO_GO_NEED_PROMPT_REBINDING"
    strong_fraction = (strong_cases / successful_cases) if successful_cases else 0.0
    if strong_fraction < min_strong_fraction:
        return "GO_WEAK_PROMPT_REBINDING"
    left_ok = _side_ok(boundary["left"], acceptable_fraction)
    right_ok = _side_ok(boundary["right"], acceptable_fraction)
    if left_ok and right_ok:
        return "STRONG_GO"
    if left_ok and not right_ok:
        return "GO_EXPECTED_RIGHT_BOUNDARY_RISK"
    return "GO" if right_ok else "GO_WITH_BOUNDARY_RISK"


def summary_cost_gate_failed(efficiency, config) -> bool:
    ratio = efficiency.get("R_time_end_to_end")
    if ratio is None:
        return False
    return ratio >= float(config["gates"]["cost"]["max_time_ratio"])


def _next_step(verdict) -> str:
    return {
        "NO_GO_STATE_INCOMPLETE": "Fix state capture before any editing work.",
        "NO_GO_NEED_PROMPT_REBINDING": "Add a light prompt-rebinding / edit-adapter post-training stage.",
        "GO_WEAK_PROMPT_REBINDING": "State reuse is proven; add light prompt-rebinding post-training.",
        "GO_EXPECTED_RIGHT_BOUNDARY_RISK": "Selective forward propagation from the edited chunk.",
        "GO_WITH_BOUNDARY_RISK": "Investigate bridge generation / local propagation before scaling edits.",
        "STRONG_GO": "Proceed to edit-ready video generation.",
    }.get(verdict, "Review the gate table before choosing the next phase.")


def gate_b_cases(records, minimum_s, minimum_ratio):
    """Per-case Gate B entry; the caller splits on ``evidence``."""
    cases = []
    for record in records:
        response = record["responsiveness"]
        local = response["text_rebind"]["S_proxy"]
        replay = response["replay"]["S_proxy"]
        control = response["crossattn_control"]["S_proxy"]
        full = response["full_regeneration"]["S_proxy"]
        # At chunk 0 the checkpoint carries no text binding yet, so the cached-K/V
        # control is not a no-op and must not veto the case.
        control_informative = record["target_chunk"] > 0
        directional = bool(local > minimum_s and local > replay
                           and (local > control or not control_informative))
        ratio = (local / full) if full > 0 else None
        cases.append({"sample_id": record["sample_id"], "prompt_id": record["prompt_id"],
                      "target_chunk": record["target_chunk"], "evidence": record.get("evidence"),
                      "control_informative": control_informative,
                      "s_proxy_text_rebind": local, "s_proxy_replay": replay,
                      "s_proxy_control": control, "s_proxy_full_regeneration": full,
                      "R_k": ratio, "passed": directional,
                      "strong": bool(directional and ratio is not None and ratio >= minimum_ratio)})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/edit_ready_mvp.yaml"))
    parser.add_argument("--replay", type=Path, nargs="*", default=None,
                        help="Experiment A JSON files (default: replay_main*.json)")
    parser.add_argument("--edit", type=Path, nargs="*", default=None,
                        help="Experiment B JSON files (default: local_edit_main*.json)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/edit_ready_mvp/summary.json")
    arguments = parser.parse_args()
    config = ex.read_config(arguments.config)
    output_dir = arguments.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_paths = arguments.replay or sorted(output_dir.glob("replay_main*.json"))
    edit_paths = arguments.edit or sorted(output_dir.glob("local_edit_main*.json"))
    collected = collect(load_many(replay_paths), load_many(edit_paths))
    replay_cases = collected["replay_cases"]
    edit_cases = collected["edit_cases"]
    # Which code actually produced the raw numbers, as recorded by the runs
    # themselves (independent of the tree state at analysis time).
    runs = collected["run_records"]
    def distinct(field):
        return sorted({record[field] for record in runs if record.get(field) is not None})
    run_provenance = {
        "files": runs,
        "git_commits": distinct("git_commit"),
        "git_dirty_values": distinct("git_dirty"),
        "code_tree_sha256": distinct("code_tree_sha256"),
        "config_sha256": distinct("config_sha256"),
        "model_checkpoint_sha256": distinct("model_hash"),
        "clean": all(record.get("git_dirty") is False for record in runs) if runs else None,
    }

    # ---- Gate A -------------------------------------------------------------
    if replay_cases:
        latencies = [case["latent"]["original_vs_replay"] for case in replay_cases]
        repeats = [case["latent"]["repeat_baseline"] for case in replay_cases]
        relative = float(config["gates"]["replay"]["relative_to_repeat"])
        epsilon = float(config["gates"]["replay"]["epsilon"])
        cosine_floor = float(config["gates"]["replay"]["cosine_floor"])
        passed_cases = sum(
            1 for value, repeat in zip(latencies, repeats)
            if value["exact"] or (value["mse"] <= relative * repeat["mse"] + epsilon
                                  and value["cosine"] >= cosine_floor))
        replay_gate = {
            "passed": passed_cases == len(replay_cases),
            "cases": len(replay_cases), "passed_cases": passed_cases,
            "exact_cases": sum(1 for value in latencies if value["exact"]),
            "rng_exact_cases": sum(1 for case in replay_cases if case["rng"].get("exact")),
            "latent_mse_mean": mean([value["mse"] for value in latencies]),
            "latent_cosine_mean": mean([value["cosine"] for value in latencies]),
            "repeat_noise_baseline": mean([value["mse"] for value in repeats]),
            "relative_to_repeat": relative, "cosine_floor": cosine_floor,
        }
    else:
        replay_gate = {"passed": None, "cases": 0, "latent_mse_mean": None,
                       "repeat_noise_baseline": None}

    # ---- Gate B -------------------------------------------------------------
    minimum_s = max(0.0, float(config["gates"]["edit"].get("min_s_proxy", 0.0)))
    minimum_ratio = float(config["gates"]["edit"].get("min_full_regeneration_ratio", 0.0))
    minimum_strong_fraction = float(config["gates"]["edit"].get("min_strong_fraction", 0.25))
    all_cases = gate_b_cases(edit_cases, minimum_s, minimum_ratio)
    directional = [case for case in all_cases if case["evidence"] == "directional"]
    # Chunk 0 has no visual history: it calibrates the rebinding implementation
    # and must not be pooled with the committed-history edit cases.
    semantic = [case for case in directional if case["target_chunk"] > 0]
    calibration = [case for case in directional if case["target_chunk"] == 0]
    qualitative = [case for case in all_cases if case["evidence"] != "directional"]
    successful = [case for case in semantic if case["passed"]]
    strong = [case for case in semantic if case["strong"]]
    by_chunk = {}
    for case in semantic:
        by_chunk.setdefault(case["target_chunk"], []).append(case["R_k"])
    edit_gate = {
        "passed": (len(successful) > 0) if semantic else None,
        "semantic_cases": len(semantic), "successful_cases": len(successful),
        "strong_cases": len(strong),
        "calibration_cases": len(calibration),
        "calibration_matches_full_regeneration": sum(
            1 for case in calibration
            if next((record["editability"].get("chunk0_matches_full_regeneration")
                     for record in edit_cases if record["sample_id"] == case["sample_id"]), None)),
        "calibration_note": "Chunk 0 has no visual history; text rebind there equals the "
                            "full-regeneration chunk bit-for-bit and calibrates R_k. These "
                            "cases are excluded from the pass/strong aggregate.",
        "min_full_regeneration_ratio": minimum_ratio,
        "R_k_by_chunk": {str(chunk): mean(values) for chunk, values in sorted(by_chunk.items())},
        "R_k_cases_by_chunk": {str(chunk): values for chunk, values in sorted(by_chunk.items())},
        "s_proxy_text_rebind_mean": mean([case["s_proxy_text_rebind"] for case in semantic]),
        "s_proxy_full_regeneration_mean": mean([case["s_proxy_full_regeneration"] for case in semantic]),
        "s_proxy_control_mean": mean([case["s_proxy_control"] for case in semantic]),
        "variants": sorted({case["prompt_id"] for case in successful}),
        "strong_variants": sorted({case["prompt_id"] for case in strong}),
        "qualitative_cases": len(qualitative),
        "qualitative_consistent_cases": sum(case["s_proxy_text_rebind"] > case["s_proxy_replay"]
                                            for case in qualitative),
        "note": "Only `evidence: directional` prompts enter the pass/strong aggregate; "
                "qualitative prompts (smile / zoom / rain) report a proxy-consistent "
                "change count and are not counted as semantic success.",
    }

    # ---- sanity controls ----------------------------------------------------
    rng_matched = [case for case in edit_cases
                   if case["sanity"].get("base_and_full_regen_share_rng_stream")]
    nonzero_targets = [case for case in edit_cases if case["target_chunk"] > 0]
    sanity = {
        "chunk0_cases": [{"sample_id": case["sample_id"],
                          "chunk0_matches_full_regeneration":
                              case["editability"].get("chunk0_matches_full_regeneration")}
                         for case in edit_cases if case["target_chunk"] == 0],
        "base_and_full_regen_share_rng_stream_cases": len(rng_matched),
        "base_and_full_regen_share_rng_stream_total": len(edit_cases),
        "control_equals_replay_cases": sum(
            1 for case in nonzero_targets if case["sanity"].get("control_equals_replay")),
        "control_equals_replay_total": len(nonzero_targets),
        "control_equals_text_rebind_at_chunk0": [
            case["sanity"].get("control_equals_text_rebind")
            for case in edit_cases if case["target_chunk"] == 0],
        "recache_equals_text_rebind_cases": sum(
            1 for case in edit_cases if case["sanity"].get("recache_equals_text_rebind")),
        "recache_cache_exact_cases": sum(
            1 for case in edit_cases if case["recache_cache_matches_checkpoint"].get("exact")),
        "recache_cache_by_chunk": {
            str(chunk): {
                "exact_cases": sum(
                    1 for case in edit_cases if case["target_chunk"] == chunk
                    and case["recache_cache_matches_checkpoint"].get("exact")),
                "cases": sum(1 for case in edit_cases if case["target_chunk"] == chunk),
                "differing_token_ranges": sorted({
                    (case["recache_cache_matches_checkpoint"].get("first_differing_token"),
                     case["recache_cache_matches_checkpoint"].get("last_differing_token"))
                    for case in edit_cases if case["target_chunk"] == chunk
                    and case["recache_cache_matches_checkpoint"].get("exact") is False}),
            } for chunk in sorted({case["target_chunk"] for case in edit_cases})},
        "recache_cache_len": next((case["recache_cache_matches_checkpoint"].get("cache_len")
                                   for case in edit_cases), None),
        "total_cases": len(edit_cases),
    }

    # ---- history confound ---------------------------------------------------
    # The reverse control replays a chunk that sits on the *P1* trajectory but is
    # asked for the P0 text.  If it lands close to the full-regeneration chunk,
    # the visual history -- not the prompt -- is what determines the reopened
    # chunk.  This is the diagnostic the plan's "history confound" question needs.
    history_rows = []
    for case in edit_cases:
        full = case["responsiveness"]["full_regeneration"]["S_proxy"]
        reverse = case["responsiveness"]["reverse"]["S_proxy"]
        history_rows.append({
            "sample_id": case["sample_id"], "target_chunk": case["target_chunk"],
            "S_reverse": reverse, "S_full_regeneration": full,
            "ratio_reverse_to_full": (reverse / full) if full > 0 else None,
            "reverse_vs_full_regeneration_mse": case["reverse_vs_full_regeneration"]["mse"],
            "text_rebind_vs_replay_mse": case["text_rebind_vs_replay"]["mse"]})
    by_chunk_history = {}
    for row in history_rows:
        by_chunk_history.setdefault(row["target_chunk"], []).append(row)
    history_confound = {
        "cases": history_rows,
        "ratio_reverse_to_full_by_chunk": {
            str(chunk): mean([row["ratio_reverse_to_full"] for row in rows])
            for chunk, rows in sorted(by_chunk_history.items())},
        "reverse_vs_full_regeneration_mse_by_chunk": {
            str(chunk): mean([row["reverse_vs_full_regeneration_mse"] for row in rows])
            for chunk, rows in sorted(by_chunk_history.items())},
        "note": "reverse = P1 history + P0 text.  A ratio near 1 means the reopened "
                "chunk follows the visual history rather than the prompt.",
    }

    # ---- Gate C -------------------------------------------------------------
    preservation_gate = {
        "passed": all(case["preservation"]["outside_exact"] for case in edit_cases)
        if edit_cases else None,
        "outside_exact_fraction": (sum(1 for case in edit_cases if case["preservation"]["outside_exact"])
                                   / len(edit_cases)) if edit_cases else None,
        "outside_max_abs_after_vae_mean": mean([case["preservation"]["outside_max_abs_after_vae"]
                                                for case in edit_cases]),
        "note": "Asserted with torch.equal on pre-decode latents; the VAE leak is "
                "reported separately and is not part of the gate.",
    }

    # ---- boundary / efficiency ---------------------------------------------
    boundary = boundary_verdict(edit_cases, config["gates"].get("boundary", {}))

    def cost(key):
        return [case["cost"].get(key) for case in edit_cases]

    efficiency = {
        "full_seconds_mean": mean(cost("full_generation_seconds")),
        "partial_generation_seconds_mean": mean(cost("partial_generation_seconds")),
        "partial_end_to_end_seconds_mean": mean(cost("partial_end_to_end_seconds")),
        "disk_load_seconds_mean": mean(cost("disk_load_seconds")),
        "host_to_device_seconds_mean": mean(cost("host_to_device_seconds")),
        "restore_seconds_mean": mean(cost("restore_seconds")),
        "encode_seconds_mean": mean(cost("encode_seconds")),
        "denoise_seconds_mean": mean(cost("denoise_seconds")),
        "R_time_generation": mean(cost("R_time_generation")),
        "R_time_compute": mean(cost("R_time_compute")),
        "R_time_end_to_end": mean(cost("R_time_end_to_end")),
        # Same user path but with the checkpoint already resident (RAM/staging),
        # i.e. without the cold storage read.  Derived from the per-case timings,
        # so no extra run is needed.
        "R_time_end_to_end_warm_cache": mean([
            ((case["cost"]["partial_end_to_end_seconds"] - case["cost"]["disk_load_seconds"])
             / case["cost"]["full_end_to_end_seconds"])
            if case["cost"].get("full_end_to_end_seconds") else None
            for case in edit_cases]),
        "cache_read_seconds_per_gib": mean([
            (case["cost"]["disk_load_seconds"] / (case["cost"]["cache_disk_bytes"] / 2 ** 30))
            if case["cost"].get("cache_disk_bytes") else None for case in edit_cases]),
        "partial_regenerated_chunks": 1,
        "full_generated_chunks_mean": mean([case["num_chunks"] for case in edit_cases]),
        "peak_vram_bytes_max": max([case["cost"]["peak_vram_bytes"] for case in edit_cases],
                                   default=None),
        "cache_bytes_per_chunk": edit_cases[0]["cost"]["cache_bytes_per_chunk"] if edit_cases else None,
        "page_cache_dropped": all(case["cost"].get("page_cache_dropped", False) for case in edit_cases)
        if edit_cases else None,
    }

    all_passed = [replay_gate["passed"], edit_gate["passed"], preservation_gate["passed"]]
    status = "passed" if all(value is True for value in all_passed) else (
        "incomplete" if any(value is None for value in all_passed) else "failed")
    acceptable_fraction = float(config["gates"].get("boundary", {}).get("acceptable_fraction", 0.9))
    verdict = decision(replay_gate["passed"] is True, edit_gate["passed"] is True,
                       edit_gate["strong_cases"], edit_gate["successful_cases"], boundary,
                       efficiency, acceptable_fraction, minimum_strong_fraction)
    notes = []
    if edit_gate["successful_cases"]:
        ratio = edit_gate["strong_cases"] / edit_gate["successful_cases"]
        if ratio < minimum_strong_fraction:
            notes.append("Prompt re-binding is directionally correct but weak: only "
                         f"{edit_gate['strong_cases']}/{edit_gate['successful_cases']} "
                         "directional cases recover >=25% of the full-regeneration response.")
    if efficiency["cache_bytes_per_chunk"] and efficiency["cache_bytes_per_chunk"] > 2 ** 30:
        notes.append("The first-version edit cache is heavy "
                     f"({efficiency['cache_bytes_per_chunk'] / 2 ** 30:.2f} GiB per chunk) and is a "
                     "*sufficient* recoverable state, not a proven minimal one.")
    notes.append("Outside preservation is bit-exact on pre-decode latents; the causal VAE "
                 "still leaks a small, decaying change into later decoded frames.")
    notes.append("Gate B covers only the directional-probe prompts; smile / zoom / rain are "
                 "reported as qualitative and need human review.")
    if summary_cost_gate_failed(efficiency, config):
        notes.append("Gate D fails on the cold-storage path: reading one ~1 GiB checkpoint "
                     "costs about as much as generating several chunks. The device-side "
                     "compute ratio is much lower, so the cost problem is cache size and I/O, "
                     "not recomputation.")

    summary = {
        "status": status,
        "run_provenance": run_provenance,
        "replay_gate": replay_gate,
        "edit_gate": edit_gate,
        "sanity_controls": sanity,
        "history_confound": history_confound,
        "preservation_gate": preservation_gate,
        "boundary": {
            "left_delta_mean": boundary["left"].get("delta_mean"),
            "left_base_mean": boundary["left"].get("base_mean"),
            "left_edited_mean": boundary["left"].get("edited_mean"),
            "left_relative_increase_of_means": boundary["left"].get("relative_increase_of_means"),
            "left_acceptable_cases": boundary["left"].get("acceptable_cases"),
            "right_delta_mean": boundary["right"].get("delta_mean"),
            "right_base_mean": boundary["right"].get("base_mean"),
            "right_edited_mean": boundary["right"].get("edited_mean"),
            "right_relative_increase_of_means": boundary["right"].get("relative_increase_of_means"),
            "right_acceptable_cases": boundary["right"].get("acceptable_cases"),
            "total_cases": boundary["right"].get("total_cases"),
            "left_acceptable": boundary["left"].get("acceptable"),
            "right_acceptable": boundary["right"].get("acceptable"),
        },
        "efficiency": efficiency,
        "gate_d_cost": {
            "passed": all(case["cost"]["R_time_end_to_end"]
                          < float(config["gates"]["cost"]["max_time_ratio"])
                          for case in edit_cases
                          if case["num_chunks"] >= int(config["gates"]["cost"]["min_chunks"]))
            if edit_cases else None,
            "passed_warm_cache": all(
                ((case["cost"]["partial_end_to_end_seconds"] - case["cost"]["disk_load_seconds"])
                 / case["cost"]["full_end_to_end_seconds"])
                < float(config["gates"]["cost"]["max_time_ratio"])
                for case in edit_cases
                if case["num_chunks"] >= int(config["gates"]["cost"]["min_chunks"])
                and case["cost"].get("full_end_to_end_seconds"))
            if edit_cases else None,
            "max_time_ratio": float(config["gates"]["cost"]["max_time_ratio"]),
            "cold_read_note": "Gate D is evaluated on the full user path including a cold "
                              "page-cache read of the ~1 GiB checkpoint; the device-side "
                              "compute ratio and the warm-cache ratio are reported separately.",
        },
        "decision": verdict,
        "decision_notes": notes,
        "next_step": _next_step(verdict),
        "limitations": [
            "Edit responsiveness uses interpretable proxies (colour fraction / luma / "
            "appearance change); no CLIP-style text-video alignment was computed because "
            "the host has no network access to fetch CLIP weights.",
            "Only local appearance / transient edits are tested; no topology, object "
            "deletion or persistent semantic change (plan section 3).",
        ],
    }
    ex.write_json(arguments.output, summary)

    timing = {
        "provenance": ex.run_provenance(config, arguments.config,
                                        extra={"replay_inputs": [str(p) for p in replay_paths],
                                               "edit_inputs": [str(p) for p in edit_paths]}),
        "cases": [{"sample_id": case["sample_id"], "target_chunk": case["target_chunk"],
                   "num_chunks": case["num_chunks"], **case["cost"]} for case in edit_cases],
    }
    ex.write_json(output_dir / "timing.json", timing)

    manifest = {
        "schema_version": 1,
        "cache_bytes_per_chunk": efficiency["cache_bytes_per_chunk"],
        "entries": [{"sample_id": case["sample_id"], "chunk_index": case["target_chunk"],
                     "cache_bytes": case["cost"]["cache_bytes_per_chunk"],
                     "disk_bytes": case["cost"]["cache_disk_bytes"],
                     "sha256": case["cache"].get("sha256"), "path": case["cache"].get("path")}
                    for case in edit_cases],
    }
    ex.write_json(output_dir / "cache_manifest.json", manifest)

    print(json.dumps({key: summary[key] for key in
                      ("status", "replay_gate", "edit_gate", "sanity_controls",
                       "history_confound", "preservation_gate", "boundary",
                       "efficiency", "decision")}, indent=2))


if __name__ == "__main__":
    main()
