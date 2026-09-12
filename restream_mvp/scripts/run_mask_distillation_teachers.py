#!/usr/bin/env python3
"""Generate new lambda=.2 Oracle teachers for M1-A distillation."""
from __future__ import annotations

import argparse
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
from restream import edit_replay as er  # noqa: E402
from restream.history_gate import history_gate  # noqa: E402
from restream.mask_distillation import (  # noqa: E402
    PROTOCOL, checkpoint_state_feature, feature_digest, prompt_delta_feature)
from restream.mask_distillation_protocol import NEW_TEACHER_SEEDS, groups_from_manifest  # noqa: E402
from restream.runtime import load_pipeline  # noqa: E402
from scripts.run_history_component_screen import validate_model_geometry  # noqa: E402
from scripts.run_oracle_layer_mask import (  # noqa: E402
    _candidate, cuda_kernel_invariants, optimize_mask)


def run_unit(pipeline, config, group, device, cache_dir, identity, oracle_plan):
    target, seed, block = 4, group["seed"], int(pipeline.num_frame_per_block)
    generation_args = dict(seed=seed, model_hash=identity["model_checkpoint_sha256"],
                           model_record=identity["model_identity"],
                           config_digest=identity["config_hash"], noise_source="global")
    group_id = f"teacher_{group['prompt_id']}_seed{seed}"
    with history_gate(pipeline, 1.0):
        base = er.stream_generate(pipeline, ex.sample_noise(config, device, seed),
                                  group["base_prompt"], sample_id=group_id,
                                  cache_dir=cache_dir, save_chunks=[target], **generation_args)
        regenerated = er.stream_generate(pipeline, ex.sample_noise(config, device, seed),
                                         group["edit_prompt"], sample_id=group_id + "_fullregen",
                                         save_chunks=[], **generation_args)
    pixels_base, pixels_full = er.decode_latents(pipeline, base.latents), er.decode_latents(pipeline, regenerated.latents)
    span = er.chunk_frame_slice(target, block, pixels_base.shape[1])
    sl = slice(span["pixel_start"], span["pixel_end"])
    base_frames = pixels_base[0, sl]
    full_score = em.evaluate_probe(group["probe"], base_frames, pixels_full[0, sl])["S_proxy"]
    entry = base.checkpoint_entries[target]

    def evaluate(mask):
        return _candidate(pipeline, entry, base, pixels_base, base_frames, group, mask, device,
                          identity, full_score, detailed=False, dino=None)

    initial = evaluate([0.5] * 30)
    optimized = optimize_mask(evaluate, oracle_plan, group, 1, initial)
    selected = evaluate(optimized["mask"])
    checkpoint = ec.load_edit_checkpoint(entry["path"], verify_sha256=entry["sha256"])
    prompt_feature = prompt_delta_feature(pipeline, group["base_prompt"], group["edit_prompt"])
    state_feature = checkpoint_state_feature(checkpoint)
    record = {"protocol": PROTOCOL, "prompt_id": group["prompt_id"], "seed": seed,
              "target_chunk": target, "lambda": 0.2, "mask": optimized["mask"],
              "best_loss": optimized["best_loss"], "trace": optimized["trace"],
              "editability": selected["editability"], "D_drift": selected["D_drift"],
              "checkpoint_sha256": entry["sha256"], "prompt_feature": prompt_feature,
              "state_feature": state_feature,
              "feature_sha256": feature_digest(prompt_feature, state_feature)}
    print(f"[teacher] {group['prompt_id']} seed={seed} E={selected['editability']['E']:+.6f} "
          f"D={selected['D_drift']:.6g}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/edit_ready_mvp.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "validation/mask_distillation_teacher_manifest.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mask_distillation_plan.json")
    parser.add_argument("--oracle-plan", type=Path, default=ROOT / "configs/oracle_layer_mask_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--teacher-approved", action="store_true")
    args = parser.parse_args()
    if not args.reviewed or not args.teacher_approved:
        parser.error("M1-A teacher generation requires explicit approval flags")
    config, manifest = ex.read_config(args.config), json.loads(args.manifest.read_text())
    plan, oracle_plan = json.loads(args.plan.read_text()), json.loads(args.oracle_plan.read_text())
    if plan.get("status") != "frozen_before_teacher_generation" or plan.get("protocol") != PROTOCOL:
        parser.error("M1-A plan is not frozen before teacher generation")
    if ec.sha256_file(args.manifest) != plan["teacher"]["new_manifest_sha256"]:
        parser.error("new teacher manifest hash mismatch")
    oracle_cfg = oracle_plan["optimization"]
    if (oracle_cfg["lambdas"][1] != plan["teacher"]["oracle_lambda"] or
            oracle_cfg["gamma_l1"] != plan["teacher"]["gamma_l1"] or
            oracle_cfg["iterations"] != 16 or oracle_cfg["algorithm"] != "SPSA"):
        parser.error("teacher optimizer differs from the frozen M0/M1 lambda=.2 protocol")
    groups = groups_from_manifest(config, manifest, "mask_distillation_teacher_m1a", NEW_TEACHER_SEEDS)
    groups = ex.shard(groups, args.shard, args.shards)
    args.cache_dir.mkdir(parents=True, exist_ok=False)
    device = ex.device_for(args.gpu)
    if device.type != "cuda":
        parser.error("teacher runner requires CUDA")
    kernel_checks = cuda_kernel_invariants(device)
    geometry = validate_model_geometry(config, {"history_partition": {
        "required_sink_size": 3, "required_num_frame_per_block": 3,
        "required_local_attention_frames": 12}})
    identity = ex.run_provenance(config, args.config, extra={
        "protocol": PROTOCOL, "stage": "M1A_oracle_teacher_generation",
        "plan_sha256": ec.sha256_file(args.plan), "manifest_sha256": ec.sha256_file(args.manifest),
        "oracle_plan_sha256": ec.sha256_file(args.oracle_plan), "history_geometry": geometry,
        "kernel_invariants": kernel_checks, "shard": args.shard, "shards": args.shards})
    pipeline = load_pipeline(config, device)
    if any(parameter.requires_grad for parameter in pipeline.generator.parameters()):
        raise AssertionError("LongLive must remain frozen")
    records = []
    try:
        for group in groups:
            records.append(run_unit(pipeline, config, group, device, args.cache_dir, identity, oracle_plan))
    except Exception as error:
        ex.write_json(args.output, {"experiment": "mask_distillation_teacher_m1a", "status": "failed",
                                   "provenance": identity, "teachers": records,
                                   "error": f"{type(error).__name__}: {error}"})
        raise
    ex.write_json(args.output, {"experiment": "mask_distillation_teacher_m1a", "schema": 1,
                               "status": "complete", "provenance": identity, "teachers": records})


if __name__ == "__main__":
    main()
