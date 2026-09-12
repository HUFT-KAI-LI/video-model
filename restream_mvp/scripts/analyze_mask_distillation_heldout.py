#!/usr/bin/env python3
"""Analyze four complete M1-A held-out shards."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream.mask_distillation_analysis import analyze  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs=4, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mask_distillation_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    records, sources, identities = [], [], []
    for path in args.results:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("experiment") != "mask_distillation_heldout_m1a":
            raise ValueError(f"incomplete held-out shard {path}")
        records.extend(payload["cases"]); sources.append({"path": str(path), "sha256": ec.sha256_file(path)})
        p = payload["provenance"]
        identities.append((p["git_commit"], p["git_dirty"], p["config_hash"],
                           p["model_checkpoint_sha256"], p["plan_sha256"],
                           p["manifest_sha256"], p["controller_sha256"]))
    if len(set(identities)) != 1 or identities[0][1]:
        raise ValueError("dirty or mismatched held-out provenance")
    if identities[0][4] != ec.sha256_file(args.plan):
        raise ValueError("held-out plan hash mismatch")
    report = analyze(records, plan, sources); report["provenance_identity"] = identities[0]
    ex.write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("decision", "passing_controllers", "pareto_wins")}, indent=2))


if __name__ == "__main__": main()
