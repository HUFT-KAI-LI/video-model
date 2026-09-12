#!/usr/bin/env python3
"""Combine and analyze D3 history attention-path result shards."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from restream import edit_cache as ec
from restream import edit_experiment as ex
from restream.history_path_analysis import analyze
from restream.history_paths import PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "configs/history_attention_path_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_d3_screen" or plan.get("protocol") != PROTOCOL:
        parser.error("D3 plan is not frozen or protocol-matched")
    plan_hash = ec.sha256_file(args.plan)
    manifest_path = ROOT / plan["manifest"]["path"]
    if ec.sha256_file(manifest_path) != plan["manifest"]["sha256"]:
        parser.error("D3 frozen manifest hash mismatch")
    records, sources = [], []
    for path in args.results:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("experiment") != "history_attention_path_d3":
            parser.error(f"incomplete or wrong result payload: {path}")
        source = payload.get("provenance", {})
        if source.get("protocol") != PROTOCOL or source.get("screen_plan_sha256") != plan_hash:
            parser.error(f"D3 protocol or plan hash mismatch: {path}")
        if source.get("manifest_sha256") != plan["manifest"]["sha256"]:
            parser.error(f"D3 manifest hash mismatch: {path}")
        checks = source.get("kernel_invariants", {})
        if source.get("git_dirty") is not False or source.get("timing_status") != "invalid_diagnostic":
            parser.error(f"dirty provenance or enabled timing: {path}")
        if not all(checks.get(name) is True for name in (
                "score_0_equals_native_current_bit_exact",
                "full_equals_native_full_bit_exact", "score_half_finite_and_active")):
            parser.error(f"D3 CUDA kernel invariants missing or failed: {path}")
        if checks.get("score_backend") != "flash_attention_2_lse":
            parser.error(f"D3 score intervention did not use frozen native FA2/LSE backend: {path}")
        identity = (source.get("git_commit"), source.get("model_checkpoint_sha256"),
                    source.get("config_hash"))
        if not all(identity):
            parser.error(f"incomplete code/model/config identity: {path}")
        sources.append({"path": str(path), "sha256": ec.sha256_file(path),
                        "git_commit": identity[0], "model_hash": identity[1],
                        "config_hash": identity[2]})
        records.extend(payload.get("cases", []))
    if len({(row["git_commit"], row["model_hash"], row["config_hash"])
            for row in sources}) != 1:
        parser.error("D3 shards do not share code/model/config identity")
    report = analyze(records, plan, sources)
    report["screen_plan"] = {"path": str(args.plan), "sha256": plan_hash,
                             "status": plan["status"]}
    report["analysis_provenance"] = ec.git_state()
    ex.write_json(args.output, report)
    print(json.dumps({"status": report["status"], "aggregate": report["aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
