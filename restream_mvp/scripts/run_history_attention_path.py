#!/usr/bin/env python3
"""D3 fixed-history score/value attention-path screen."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code/LongLive")]

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream.history_gate import history_gate  # noqa: E402
from restream.history_paths import (  # noqa: E402
    CONDITIONS, PROTOCOL, condition_spec, manifest_groups, validate_screen)
from restream.runtime import load_pipeline  # noqa: E402
from scripts.run_history_component_screen import (  # noqa: E402
    _run_fixed_group, validate_model_geometry)
def cuda_kernel_invariants(device):
    spec = importlib.util.spec_from_file_location(
        "d3_history_attention", ROOT / "code/LongLive/wan/modules/attention.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    q = torch.randn(1, 6, 2, 64, device=device, dtype=torch.bfloat16)
    kh = torch.randn(1, 9, 2, 64, device=device, dtype=torch.bfloat16)
    vh = torch.randn_like(kh)
    kc = torch.randn(1, 6, 2, 64, device=device, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    current = kernel.attention(q, kc, vc)
    score_zero = kernel.path_gated_attention(q, kh, vh, kc, vc, 0, 1)
    full = kernel.attention(q, torch.cat([kh, kc], 1), torch.cat([vh, vc], 1))
    path_full = kernel.path_gated_attention(q, kh, vh, kc, vc, 1, 1)
    score_half = kernel.path_gated_attention(q, kh, vh, kc, vc, .5, 1)
    if not torch.equal(score_zero, current):
        raise AssertionError("D3 score=0 is not bit-exact native current-only attention")
    if not torch.equal(path_full, full):
        raise AssertionError("D3 path full is not bit-exact native full attention")
    if not torch.isfinite(score_half).all() or torch.equal(score_half, full):
        raise AssertionError("D3 score=.5 CUDA path is invalid or inert")
    return {"score_0_equals_native_current_bit_exact": True,
            "full_equals_native_full_bit_exact": True,
            "score_half_finite_and_active": True,
            "score_backend": "flash_attention_2_lse"
            if kernel.FLASH_ATTN_2_AVAILABLE else "sdpa_broadcast_bias",
            "dtype": str(q.dtype), "device": str(device)}


def run_group(pipeline, config, group, device, output_dir, cache_dir, identity, dino=None):
    with history_gate(pipeline, 1.0):
        return _run_fixed_group(
            pipeline, config, group, device, output_dir, cache_dir, identity, dino,
            protocol=PROTOCOL, spec_fn=condition_spec, full_condition="full", stage="D3")


def assert_path_invariants(records):
    grouped = {}
    for record in records:
        key = (record["prompt_id"], record["seed"], record["target_chunk"])
        cases = grouped.setdefault(key, {})
        if record["condition"] in cases:
            raise AssertionError(f"{key}: duplicate condition {record['condition']}")
        cases[record["condition"]] = record
    for key, cases in grouped.items():
        if set(cases) != set(CONDITIONS) or key[2] != 4:
            raise AssertionError(f"{key}: incomplete or non-chunk4 D3 condition set")
        for name, case in cases.items():
            if condition_spec(name)["kind"] == "global":
                if case.get("history_partition") is not None:
                    raise AssertionError(f"{key}: global condition used path routing")
            else:
                audits = case.get("history_partition", {})
                if set(audits) != {"replay", "text_rebind"} or any(
                        audits[p]["latent_frames"] != {"history": 9, "current": 3}
                        for p in ("replay", "text_rebind")):
                    raise AssertionError(f"{key}: path partition audit failed for {name}")
        if not cases["full"]["sanity"].get("full_history_P0_exact_base"):
            raise AssertionError(f"{key}: full path P0 is not exact base")
        full_hashes = cases["full"]["chunk_latent_sha256"]
        if all(cases["current_only"]["chunk_latent_sha256"][p] == full_hashes[p]
               for p in ("replay", "text_rebind")):
            raise AssertionError(f"{key}: current-only is inert")
        for name in ("score_.5", "value_.5", "value_0", "score_.5_value_.5"):
            if all(cases[name]["chunk_latent_sha256"][p] == full_hashes[p]
                   for p in ("replay", "text_rebind")):
                raise AssertionError(f"{key}: {name} is inert")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/edit_ready_mvp.yaml")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "validation/history_attention_path_manifest.json")
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "configs/history_attention_path_plan.json")
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
        parser.error("D3 GPU generation requires --reviewed and --screen-approved")
    if not args.dino:
        parser.error("D3 plan requires --dino")
    config = ex.read_config(args.config)
    manifest = json.loads(args.manifest.read_text())
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_d3_screen" or plan.get("protocol") != PROTOCOL:
        parser.error("D3 plan must be frozen and protocol-matched")
    if ec.sha256_file(args.manifest) != plan["manifest"]["sha256"]:
        parser.error("D3 manifest differs from the frozen plan")
    try:
        geometry = validate_model_geometry(config, plan)
        all_groups = manifest_groups(config, manifest)
        validate_screen(all_groups, plan)
        groups = ex.shard(all_groups, args.shard, args.shards)
    except ValueError as error:
        parser.error(str(error))
    if not groups:
        parser.error("No D3 groups selected")
    root = ROOT / "validation/history_paths"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"d3_shard{args.shard}_", dir=root))
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
        parser.error("D3 runner requires working CUDA")
    torch.set_grad_enabled(False)
    kernel_checks = cuda_kernel_invariants(device)
    identity = ex.run_provenance(config, args.config, extra={
        "protocol": PROTOCOL, "stage": "D3_score_value_attention_path_decomposition",
        "manifest_path": str(args.manifest), "manifest_sha256": ec.sha256_file(args.manifest),
        "screen_plan_path": str(args.plan), "screen_plan_sha256": ec.sha256_file(args.plan),
        "screen_plan_status": plan["status"], "history_geometry": geometry,
        "conditions": plan["conditions"], "kernel_invariants": kernel_checks,
        "shard": args.shard, "shards": args.shards, "video_root": str(video_root),
        "cache_dir": str(cache_dir), "timing_status": "invalid_diagnostic"})
    pipeline = load_pipeline(config, device)
    dino = em.DinoFeatureDistance(ROOT / config["model"]["dino"], device=device)
    records = []
    try:
        for group in groups:
            records.extend(run_group(pipeline, config, group, device, video_root,
                                     cache_dir, identity, dino=dino))
        assert_path_invariants(records)
    except Exception as error:
        ex.write_json(output, {"experiment": "history_attention_path_d3", "status": "failed",
                              "provenance": identity, "cases": records,
                              "error": f"{type(error).__name__}: {error}"})
        raise
    ex.write_json(output, {"experiment": "history_attention_path_d3", "schema": 1,
                           "status": "complete", "provenance": identity, "config": config,
                           "cases": records,
                           "screen": {"passed_invariants": True, "pairs": len(records),
                                      "chunk_replays": 2 * len(records),
                                      "interpretation": "exploratory_no_significance_test"}})
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
