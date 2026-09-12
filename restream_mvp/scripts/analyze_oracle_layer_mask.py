#!/usr/bin/env python3
"""Analyze complete M0/M1 oracle layer-mask shards."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream.oracle_layer_analysis import analyze  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/oracle_layer_mask_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    records, sources, identities = [], [], []
    for path in args.results:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("experiment") != "oracle_layer_release_m0m1":
            raise ValueError(f"incomplete oracle shard: {path}")
        records.extend(payload["cases"])
        provenance = payload["provenance"]
        identities.append((provenance["git_commit"], provenance["config_hash"],
                           provenance["model_checkpoint_sha256"], provenance["plan_sha256"],
                           provenance["manifest_sha256"], provenance["git_dirty"]))
        sources.append({"path": str(path), "sha256": ec.sha256_file(path)})
    if len(set(identities)) != 1 or identities[0][-1]:
        raise ValueError("oracle shards have dirty or mismatched provenance")
    if identities[0][3] != ec.sha256_file(args.plan):
        raise ValueError("analysis plan hash differs from generation")
    report = analyze(records, plan, sources)
    report["provenance_identity"] = identities[0][:-1]
    ex.write_json(args.output, report)
    print(json.dumps({"decision": report["decision"],
                      "units_with_oracle_pareto_gain": report["units_with_oracle_pareto_gain"],
                      "total_units": report["total_units"]}, indent=2))


if __name__ == "__main__":
    main()
