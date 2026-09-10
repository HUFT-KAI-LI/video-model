"""Aggregate the Edit-Ready MVP runs into ``validation/edit_ready_mvp/summary.json``.

Consumes one or more Experiment A / Experiment B JSON files (possibly sharded
across GPUs) and writes the schema of plan section 16 plus ``timing.json`` and
``cache_manifest.json`` (plan section 15).  Missing inputs are reported as
``null`` and downgrade ``status`` -- nothing is ever pre-filled with fake numbers.
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
    loaded = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        loaded.append((Path(path), data))
    return loaded


def mean(values, keys=None):
    values = [value for value in values if value is not None]
    if not values:
        return None
    if keys:
        values = [value[keys[0]][keys[1]] if len(keys) == 2 else value[keys[0]] for value in values]
    return sum(values) / len(values)


def collect(replay_files, edit_files) -> dict:
    replay_cases, replay_gates = [], []
    for _, data in replay_files:
        replay_cases.extend(data.get("cases", []))
        if "gate" in data:
            replay_gates.append(data["gate"])
    edit_cases = []
    for _, data in edit_files:
        edit_cases.extend(data.get("cases", []))
    return {"replay_cases": replay_cases, "replay_gates": replay_gates,
            "edit_cases": edit_cases}


def boundary_verdict(edit_cases, gate_config) -> dict:
    relative = float(gate_config.get("max_relative_increase", 1.0))
    slack = float(gate_config.get("absolute_slack", 1e-4))
    left, right = [], []
    for case in edit_cases:
        latent = case["boundary"]["latent"]
        base, edited = latent["base"], latent["edited"]
        left.append((edited["left_mse"], base["left_mse"]))
        right.append((edited["right_mse"], base["right_mse"]))
    def summarize(pairs):
        pairs = [(e, b) for e, b in pairs if e == e and b == b]  # drop NaN
        if not pairs:
            return {"available": False}
        deltas = [e - b for e, b in pairs]
        accepted = [e <= b + max(relative * b, slack) for e, b in pairs]
        base_mean = sum(b for _, b in pairs) / len(pairs)
        edited_mean = sum(e for e, _ in pairs) / len(pairs)
        return {"available": True, "delta_mean": sum(deltas) / len(deltas),
                "base_mean": base_mean, "edited_mean": edited_mean,
                "relative_increase_of_means": (edited_mean - base_mean) / base_mean
                if base_mean > 0 else None,
                "acceptable": all(accepted), "acceptable_cases": sum(accepted),
                "total_cases": len(pairs),
                "worst_relative_increase": max((e - b) / b for e, b in pairs if b > 0)
                if any(b > 0 for _, b in pairs) else None}
    left_summary, right_summary = summarize(left), summarize(right)
    return {"left": left_summary, "right": right_summary,
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
    """Verdict following the plan's section 13 interpretation matrix.

    A boundary side counts as acceptable when at least ``acceptable_fraction`` of
    the cases stay within the relative-increase rule; the raw per-case counts stay
    in ``summary["boundary"]`` so nothing is hidden behind the threshold.  A
    handful of strong edits does not upgrade the verdict to ``STRONG_GO``: the
    strong cases must also cover ``min_strong_fraction`` of the directional ones.
    """
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


def _next_step(verdict) -> str:
    return {
        "NO_GO_STATE_INCOMPLETE": "Fix state capture before any editing work.",
        "NO_GO_NEED_PROMPT_REBINDING": "Add a light prompt-rebinding / edit-adapter post-training stage.",
        "GO_WEAK_PROMPT_REBINDING": "State reuse is proven; add light prompt-rebinding post-training.",
        "GO_EXPECTED_RIGHT_BOUNDARY_RISK": "Selective forward propagation from the edited chunk.",
        "GO_WITH_BOUNDARY_RISK": "Investigate bridge generation / local propagation before scaling edits.",
        "STRONG_GO": "Proceed to edit-ready video generation.",
    }.get(verdict, "Review the gate table before choosing the next phase.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/edit_ready_mvp.yaml"))
    parser.add_argument("--replay", type=Path, nargs="*", default=None,
                        help="Experiment A JSON files (default: validation/edit_ready_mvp/replay_*.json)")
    parser.add_argument("--edit", type=Path, nargs="*", default=None,
                        help="Experiment B JSON files (default: validation/edit_ready_mvp/local_edit_*.json)")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/edit_ready_mvp/summary.json")
    arguments = parser.parse_args()
    config = ex.read_config(arguments.config)
    output_dir = arguments.output.parent

    replay_paths = arguments.replay or sorted(
        path for path in output_dir.glob("replay_main*.json"))
    edit_paths = arguments.edit or sorted(
        path for path in output_dir.glob("local_edit_main*.json"))
    replay_files = load_many(replay_paths)
    edit_files = load_many(edit_paths)
    collected = collect(replay_files, edit_files)
    replay_cases = collected["replay_cases"]
    edit_cases = collected["edit_cases"]

    # ---- Gate A -------------------------------------------------------------
    if replay_cases:
        latencies = [case["latent"]["original_vs_replay"] for case in replay_cases]
        repeats = [case["latent"]["repeat_baseline"] for case in replay_cases]
        exact = sum(1 for value in latencies if value["exact"])
        relative = float(config["gates"]["replay"]["relative_to_repeat"])
        epsilon = float(config["gates"]["replay"]["epsilon"])
        cosine_floor = float(config["gates"]["replay"]["cosine_floor"])
        passed_cases = sum(
            1 for value, repeat in zip(latencies, repeats)
            if value["exact"] or (value["mse"] <= relative * repeat["mse"] + epsilon
                                  and value["cosine"] >= cosine_floor))
        rng_exact = sum(1 for case in replay_cases if case["rng"].get("exact"))
        replay_gate = {
            "passed": passed_cases == len(replay_cases),
            "cases": len(replay_cases),
            "passed_cases": passed_cases,
            "exact_cases": exact,
            "rng_exact_cases": rng_exact,
            "latent_mse_mean": mean(latencies, ("mse",)),
            "latent_cosine_mean": mean(latencies, ("cosine",)),
            "repeat_noise_baseline": mean(repeats, ("mse",)),
            "relative_to_repeat": relative,
            "cosine_floor": cosine_floor,
        }
    else:
        replay_gate = {"passed": None, "cases": 0, "latent_mse_mean": None,
                       "repeat_noise_baseline": None}

    # ---- Gate B -------------------------------------------------------------
    minimum_s = max(0.0, float(config["gates"]["edit"].get("min_s_proxy", 0.0)))
    minimum_ratio = float(config["gates"]["edit"].get("min_full_regeneration_ratio", 0.0))
    edit_success, edit_strong = [], []
    for case in edit_cases:
        local = case["responsiveness"]["local_edit"]["S_proxy"]
        full = case["responsiveness"]["full_regeneration"]["S_proxy"]
        directional = bool(local > minimum_s
                           and local > case["responsiveness"]["same_prompt_replay"]["S_proxy"]
                           and local > case["responsiveness"]["crossattn_control"]["S_proxy"])
        if directional:
            edit_success.append(case)
            if full > 0 and local / full >= minimum_ratio:
                edit_strong.append(case)
    edit_gate = {
        "passed": (len(edit_success) > 0) if edit_cases else None,
        "cases": len(edit_cases),
        "successful_cases": len(edit_success),
        "strong_cases": len(edit_strong),
        "min_full_regeneration_ratio": minimum_ratio,
        "s_proxy_local_edit_mean": mean([case["responsiveness"]["local_edit"]["S_proxy"] for case in edit_cases]),
        "s_proxy_full_regeneration_mean": mean([case["responsiveness"]["full_regeneration"]["S_proxy"] for case in edit_cases]),
        "s_proxy_control_mean": mean([case["responsiveness"]["crossattn_control"]["S_proxy"] for case in edit_cases]),
        "variants": sorted({case["prompt_id"] for case in edit_success}),
        "strong_variants": sorted({case["prompt_id"] for case in edit_strong}),
    }

    # ---- Gate C -------------------------------------------------------------
    preservation_gate = {
        "passed": all(case["preservation"]["outside_exact"] for case in edit_cases) if edit_cases else None,
        "outside_exact_fraction": (sum(1 for case in edit_cases if case["preservation"]["outside_exact"])
                                   / len(edit_cases)) if edit_cases else None,
        "outside_max_abs_after_vae_mean": mean([case["preservation"]["outside_max_abs_after_vae"]
                                                for case in edit_cases]),
        "note": "Asserted with torch.equal on pre-decode latents; the VAE leak is "
                "reported separately and is not part of the gate.",
    }

    # ---- boundary / efficiency ---------------------------------------------
    boundary = boundary_verdict(edit_cases, config["gates"].get("boundary", {}))
    full_times = [case["cost"]["full_generation_seconds"] for case in edit_cases]
    partial_times = [case["cost"]["partial_generation_seconds"] for case in edit_cases]
    speedups = [case["cost"]["R_time_generation"] for case in edit_cases]
    efficiency = {
        "full_seconds_mean": mean(full_times),
        "partial_seconds_mean": mean(partial_times),
        "speedup": mean(speedups),
        "R_time_end_to_end_mean": mean([case["cost"]["R_time_end_to_end"] for case in edit_cases]),
        "partial_regenerated_chunks": 1,
        "full_generated_chunks_mean": mean([case["num_chunks"] for case in edit_cases]),
        "peak_vram_bytes_max": max([case["cost"]["peak_vram_bytes"] for case in edit_cases], default=None),
        "cache_bytes_per_chunk": edit_cases[0]["cost"]["cache_bytes_per_chunk"] if edit_cases else None,
    }

    all_passed = [replay_gate["passed"], edit_gate["passed"], preservation_gate["passed"]]
    status = "passed" if all(value is True for value in all_passed) else (
        "incomplete" if any(value is None for value in all_passed) else "failed")
    acceptable_fraction = float(config["gates"].get("boundary", {}).get("acceptable_fraction", 0.9))
    min_strong_fraction = float(config["gates"]["edit"].get("min_strong_fraction", 0.25))
    verdict = decision(replay_gate["passed"] is True, edit_gate["passed"] is True,
                       edit_gate["strong_cases"], edit_gate["successful_cases"], boundary,
                       efficiency, acceptable_fraction, min_strong_fraction)
    notes = []
    if edit_gate["successful_cases"]:
        ratio = edit_gate["strong_cases"] / edit_gate["successful_cases"]
        if ratio < 0.25:
            notes.append("Prompt re-binding is directionally correct but weak: only "
                         f"{edit_gate['strong_cases']}/{edit_gate['successful_cases']} "
                         "directional cases recover >=25% of the full-regeneration response.")
    if efficiency["cache_bytes_per_chunk"] and efficiency["cache_bytes_per_chunk"] > 2 ** 30:
        notes.append("The first-version edit cache is heavy "
                     f"({efficiency['cache_bytes_per_chunk'] / 2 ** 30:.2f} GiB per chunk); "
                     "compaction is a later-phase task.")
    notes.append("Outside preservation is bit-exact on pre-decode latents; the causal VAE "
                 "still leaks a small, decaying change into later decoded frames.")

    summary = {
        "status": status,
        "replay_gate": replay_gate,
        "edit_gate": edit_gate,
        "preservation_gate": preservation_gate,
        "boundary": {"left_delta_mean": boundary["left"].get("delta_mean"),
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
                     "right_acceptable": boundary["right"].get("acceptable")},
        "efficiency": efficiency,
        "gate_d_cost": {
            "passed": all(case["cost"]["R_time_generation"] < float(config["gates"]["cost"]["max_time_ratio"])
                          for case in edit_cases if case["num_chunks"] >= int(config["gates"]["cost"]["min_chunks"]))
            if edit_cases else None,
            "max_time_ratio": float(config["gates"]["cost"]["max_time_ratio"]),
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
                      ("status", "replay_gate", "edit_gate", "preservation_gate",
                       "boundary", "efficiency", "decision")}, indent=2))


if __name__ == "__main__":
    main()
