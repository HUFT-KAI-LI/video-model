"""Target-level summary of a completed paired counterfactual probe (no model runs).

Reads a probe report written by ``reality_paired_diagnostics.probe_paired``
(directly or through ``scripts/probe_existing_memory_checkpoint.py``) and reports,
per prefix mode:

  * the mean video loss of every control variant;
  * the core paired deltas ``U_correct``, ``G_branch``, ``G_generic``,
    ``G_content`` and ``S_reference``, each first paired inside a target (same
    target, history and noise) and then aggregated over UNIQUE TARGETS;
  * a target-level bootstrap 95% confidence interval and the number of targets
    with a positive delta.

Noise seeds are repeated measures inside a target and are never treated as
independent samples. The bootstrap resamples targets, not target-noise pairs.
"""
import argparse
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json
from restream.reality_paired_diagnostics import CONTROL_KINDS, core_delta_spec
from restream.reality_stats import bootstrap_ci

BOOTSTRAP_SAMPLES = 10000


def target_entries(cases, mode, correct_kind="async"):
    """One entry per unique target: per-kind mean over its noise seeds and deltas."""
    spec = core_delta_spec(correct_kind)
    rows = [case for case in cases if case["prefix_mode"] == mode]
    entries = []
    for sample_id in dict.fromkeys(case["sample_id"] for case in rows):
        selected = [case for case in rows if case["sample_id"] == sample_id]
        means = {}
        for kind in CONTROL_KINDS:
            values = [case["variants"][kind]["video_loss"] for case in selected
                      if kind in case["variants"] and case["variants"][kind].get("video_loss") is not None]
            if values:
                means[kind] = statistics.mean(values)
        deltas = {name: (means[a] - means[b]) if a in means and b in means else None
                  for name, (a, b) in spec.items()}
        entries.append({"sample_id": sample_id, "source_id": selected[0].get("source_id"),
                        "noise_seeds": sorted({case.get("noise_seed") for case in selected}),
                        **means, "core_deltas": deltas})
    return entries


def summarize(report, samples=BOOTSTRAP_SAMPLES, seed=0):
    config = report.get("config") or {}
    correct_kind = ((config.get("reality_memory", {}).get("objective", {}) or {}).get("paired") or {}).get("correct_kind", "async")
    spec = core_delta_spec(correct_kind)
    result = {"split": report.get("split"), "correct_kind": correct_kind,
              "global_constant": report.get("global_constant"),
              "scope": report.get("scope"), "bootstrap_samples": samples, "modes": {}}
    for mode in ("clean", "mild"):
        entries = target_entries(report["cases"], mode, correct_kind)
        variants = {}
        for kind in CONTROL_KINDS:
            values = [entry[kind] for entry in entries if kind in entry]
            if values:
                variants[kind] = {"mean": statistics.mean(values), "n": len(values)}
        deltas = {name: bootstrap_ci([entry["core_deltas"][name] for entry in entries
                                      if entry["core_deltas"][name] is not None], samples, seed)
                  for name in spec}
        result["modes"][mode] = {"unique_targets": len(entries), "variants": variants,
                                 "core_deltas": deltas, "targets": entries}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True, help="Probe report JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = json.loads(args.probe.read_text())
    summary = summarize(report, args.bootstrap_samples, args.seed)
    write_json(args.output, summary)
    for mode, entry in summary["modes"].items():
        print(f"[{summary['split']}/{mode}] unique targets={entry['unique_targets']}")
        for name, value in entry["core_deltas"].items():
            print(f"  {name:12s} mean={value['mean'] if value['mean'] is None else round(value['mean'], 6)} "
                  f"95%CI=[{value['low']}, {value['high']}] positive={value['positive']}/{value['n']}")


if __name__ == "__main__":
    main()
