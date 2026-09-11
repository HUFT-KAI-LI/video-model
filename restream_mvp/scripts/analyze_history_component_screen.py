#!/usr/bin/env python3
"""Combine and analyze D1 history-component result shards."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restream import edit_cache as ec
from restream import edit_experiment as ex
from restream.history_component_analysis import analyze
from restream.history_components import PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "configs/history_component_screen_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_d1_screen" or plan.get("protocol") != PROTOCOL:
        parser.error("D1 analysis plan is not frozen or protocol-matched")
    plan_hash = ec.sha256_file(args.plan)
    manifest_path = ROOT / plan["manifest"]["path"]
    if ec.sha256_file(manifest_path) != plan["manifest"]["sha256"]:
        parser.error("D1 frozen manifest hash mismatch")
    records, provenance = [], []
    for path in args.results:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("experiment") != "history_component_screen_d1":
            parser.error(f"incomplete or wrong result payload: {path}")
        source = payload.get("provenance", {})
        if source.get("protocol") != PROTOCOL:
            parser.error(f"D1 provenance protocol mismatch: {path}")
        if source.get("screen_plan_sha256") != plan_hash:
            parser.error(f"D1 plan hash mismatch: {path}")
        if source.get("manifest_sha256") != plan["manifest"]["sha256"]:
            parser.error(f"D1 manifest hash mismatch: {path}")
        if source.get("git_dirty") is not False:
            parser.error(f"run provenance is dirty or missing cleanliness: {path}")
        if source.get("timing_status") != "invalid_diagnostic":
            parser.error(f"timing was not disabled: {path}")
        required_identity = (source.get("git_commit"), source.get("model_checkpoint_sha256"),
                             source.get("config_hash"))
        if not all(required_identity):
            parser.error(f"incomplete code/model/config identity: {path}")
        provenance.append({"path": str(path), "sha256": ec.sha256_file(path),
                           "git_commit": source.get("git_commit"),
                           "model_hash": source.get("model_checkpoint_sha256"),
                           "config_hash": source.get("config_hash")})
        records.extend(payload.get("cases", []))
    identities = {(row["git_commit"], row["model_hash"], row["config_hash"])
                  for row in provenance}
    if len(identities) != 1:
        parser.error("D1 shards do not share code/model/config identity")
    report = analyze(records, plan)
    report["screen_plan"] = {"path": str(args.plan), "sha256": plan_hash,
                             "status": plan["status"]}
    report["sources"] = provenance
    report["analysis_provenance"] = ec.git_state()
    ex.write_json(args.output, report)
    print(json.dumps({"invariants_passed": report["invariants_passed"],
                      "frontier_conditions": report["pareto"]["frontier_conditions"],
                      "selective_dominates_global": [
                          row["condition"] for row in report["pareto"]["selective_dominates_global"]
                          if row["aggregate_dominates_global"]]}, indent=2))


if __name__ == "__main__":
    main()
