#!/usr/bin/env python3
"""D2 fixed-history binary subset interaction screen."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream.history_gate import history_gate  # noqa: E402
from restream.history_subsets import (  # noqa: E402
    CONDITIONS, PROTOCOL, condition_spec, manifest_groups, validate_screen)
from restream.runtime import load_pipeline  # noqa: E402
from scripts.run_history_component_screen import (  # noqa: E402
    _run_fixed_group, validate_model_geometry)

ROOT = Path(__file__).resolve().parents[1]


def run_group(pipeline, config, group, device, output_dir, cache_dir, identity, dino=None):
    """Generate one full-history trajectory and replay all nine D2 conditions."""
    with history_gate(pipeline, 1.0):
        return _run_fixed_group(
            pipeline, config, group, device, output_dir, cache_dir, identity, dino,
            protocol=PROTOCOL, spec_fn=condition_spec, full_condition="SOR", stage="D2")


def assert_subset_invariants(records):
    grouped = {}
    for record in records:
        key = (record["prompt_id"], record["seed"], record["target_chunk"])
        cases = grouped.setdefault(key, {})
        if record["condition"] in cases:
            raise AssertionError(f"{key}: duplicate condition {record['condition']}")
        cases[record["condition"]] = record
    expected_partition = {"sink": 3, "old": 3, "recent": 3, "current": 3}
    for key, cases in grouped.items():
        if set(cases) != set(CONDITIONS):
            raise AssertionError(f"{key}: incomplete D2 condition set")
        if key[2] != 4:
            raise AssertionError(f"{key}: D2 main screen is chunk 4 only")
        for condition, case in cases.items():
            if condition == "global_release":
                if case.get("history_partition") is not None:
                    raise AssertionError(f"{key}: global reference used component routing")
                continue
            partitions = case.get("history_partition", {})
            if set(partitions) != {"replay", "text_rebind"}:
                raise AssertionError(f"{key}: missing partition audit for {condition}")
            if any(partitions[policy]["latent_frames"] != expected_partition
                   for policy in ("replay", "text_rebind")):
                raise AssertionError(f"{key}: unexpected partition for {condition}")
        if not cases["SOR"]["sanity"].get("full_history_P0_exact_base"):
            raise AssertionError(f"{key}: SOR P0 is not exact base")
        if all(cases["empty"]["chunk_latent_sha256"][policy] ==
               cases["SOR"]["chunk_latent_sha256"][policy]
               for policy in ("replay", "text_rebind")):
            raise AssertionError(f"{key}: current-only endpoint is inert")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/edit_ready_mvp.yaml")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "validation/history_subset_interaction_manifest.json")
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "configs/history_subset_interaction_plan.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dino", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--screen-approved", action="store_true")
    args = parser.parse_args()
    if not args.reviewed or not args.screen_approved:
        parser.error("D2 GPU generation requires --reviewed and --screen-approved")
    if not args.dino:
        parser.error("D2 plan requires --dino")
    config = ex.read_config(args.config)
    manifest = json.loads(args.manifest.read_text())
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_d2_screen" or plan.get("protocol") != PROTOCOL:
        parser.error("D2 plan must be frozen and protocol-matched")
    if ec.sha256_file(args.manifest) != plan["manifest"]["sha256"]:
        parser.error("D2 manifest differs from the frozen plan")
    try:
        geometry = validate_model_geometry(config, plan)
        all_groups = manifest_groups(config, manifest)
        validate_screen(all_groups, plan)
        groups = ex.shard(all_groups, args.shard, args.shards)
    except ValueError as error:
        parser.error(str(error))
    if not groups:
        parser.error("No D2 groups selected")
    root = ROOT / "validation/history_subsets"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"d2_shard{args.shard}_", dir=root))
    output = args.output or run_dir / "results.json"
    video_root = args.video_root or run_dir / "videos"
    cache_dir = args.cache_dir or run_dir / "cache"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    for path in (video_root, cache_dir):
        if path.exists() and any(path.iterdir()):
            parser.error(f"Output directory must be empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = ex.device_for(args.gpu)
    if device.type != "cuda":
        parser.error("D2 runner requires working CUDA")
    torch.set_grad_enabled(False)
    identity = ex.run_provenance(config, args.config, extra={
        "protocol": PROTOCOL, "stage": "D2_history_subset_interactions",
        "manifest_path": str(args.manifest), "manifest_sha256": ec.sha256_file(args.manifest),
        "screen_plan_path": str(args.plan), "screen_plan_sha256": ec.sha256_file(args.plan),
        "screen_plan_status": plan["status"], "history_geometry": geometry,
        "conditions": plan["conditions"], "shard": args.shard, "shards": args.shards,
        "video_root": str(video_root), "cache_dir": str(cache_dir),
        "timing_status": "invalid_diagnostic"})
    pipeline = load_pipeline(config, device)
    dino = em.DinoFeatureDistance(ROOT / config["model"]["dino"], device=device)
    records = []
    try:
        for group in groups:
            records.extend(run_group(pipeline, config, group, device, video_root,
                                     cache_dir, identity, dino=dino))
        assert_subset_invariants(records)
    except Exception as error:
        ex.write_json(output, {"experiment": "history_subset_interaction_d2",
                              "status": "failed", "provenance": identity,
                              "cases": records, "error": f"{type(error).__name__}: {error}"})
        raise
    ex.write_json(output, {"experiment": "history_subset_interaction_d2", "schema": 1,
                           "status": "complete", "provenance": identity, "config": config,
                           "cases": records,
                           "screen": {"passed_invariants": True, "pairs": len(records),
                                      "chunk_replays": 2 * len(records),
                                      "interpretation": "exploratory_no_significance_test"}})
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
