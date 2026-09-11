#!/usr/bin/env python3
"""Apply the frozen paired/Pareto plan to completed sweep JSON shards."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restream import edit_cache as ec
from restream import edit_experiment as ex
from restream.history_release_analysis import analyze

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "configs/history_release_analysis_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_full_sweep":
        parser.error("analysis plan is not frozen")
    plan_hash = ec.sha256_file(args.plan)
    manifest_path = ROOT / plan["manifest"]["path"]
    if ec.sha256_file(manifest_path) != plan["manifest"]["sha256"]:
        parser.error("frozen sweep manifest hash mismatch")
    records, provenance = [], []
    for path in args.results:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete":
            parser.error(f"incomplete result payload: {path}")
        source = payload.get("provenance", {})
        if source.get("analysis_plan_sha256") != plan_hash:
            parser.error(f"analysis plan hash mismatch: {path}")
        if source.get("git_dirty"):
            parser.error(f"dirty run provenance: {path}")
        if payload.get("gates", {}).get("gate_d_cost", {}).get("status") != "invalid_diagnostic":
            parser.error(f"Gate D was not disabled: {path}")
        provenance.append({"path": str(path), "sha256": ec.sha256_file(path),
                           "git_commit": source.get("git_commit"),
                           "model_hash": source.get("model_checkpoint_sha256"),
                           "config_hash": source.get("config_hash")})
        records.extend(payload.get("cases", []))
    identities = {(row["git_commit"], row["model_hash"], row["config_hash"])
                  for row in provenance}
    if len(identities) != 1:
        parser.error("result shards do not share code/model/config identity")
    report = analyze(records, plan)
    report["analysis_plan"] = {"path": str(args.plan), "sha256": plan_hash,
                               "status": plan["status"]}
    report["sources"] = provenance
    ex.write_json(args.output, report)
    print(json.dumps({"confirmatory": report["confirmatory"],
                      "pareto_frontier": report["pareto"]["frontier_gates"]}, indent=2))


if __name__ == "__main__":
    main()
