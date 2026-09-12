#!/usr/bin/env python3
"""Evaluate distilled controllers and Oracle upper bound on held-out M1-A seeds."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code/LongLive")]

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_media as emedia  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.history_gate import history_gate  # noqa: E402
from restream.mask_distillation import (  # noqa: E402
    PROTOCOL, checkpoint_state_feature, feature_digest, load_controllers, normalize,
    prompt_delta_feature)
from restream.mask_distillation_protocol import (  # noqa: E402
    FINAL_CONDITIONS, HELDOUT_SEEDS, groups_from_manifest)
from restream.runtime import load_pipeline  # noqa: E402
from scripts.run_history_component_screen import validate_model_geometry  # noqa: E402
from scripts.run_oracle_layer_mask import (  # noqa: E402
    _candidate, cuda_kernel_invariants, optimize_mask)


def predict_masks(controllers, prompt, state):
    norm = controllers["normalization"]
    prompt_n = normalize(prompt, norm["prompt_mean"], norm["prompt_std"]).unsqueeze(0)
    state_n = normalize(state, norm["state_mean"], norm["state_std"]).unsqueeze(0)
    with torch.no_grad():
        return {"prompt_only": controllers["prompt_only"](prompt_n)[0].tolist(),
                "prompt_state": controllers["prompt_state"](prompt_n, state_n)[0].tolist()}


def run_unit(pipeline, config, group, device, cache_dir, video_root, identity, plan,
             oracle_plan, controllers, dino):
    target, seed, block = 4, group["seed"], int(pipeline.num_frame_per_block)
    generation_args = dict(seed=seed, model_hash=identity["model_checkpoint_sha256"],
                           model_record=identity["model_identity"],
                           config_digest=identity["config_hash"], noise_source="global")
    group_id = f"heldout_{group['prompt_id']}_seed{seed}"
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
    checkpoint = ec.load_edit_checkpoint(entry["path"], verify_sha256=entry["sha256"])
    prompt = prompt_delta_feature(pipeline, group["base_prompt"], group["edit_prompt"])
    state = checkpoint_state_feature(checkpoint)
    masks = predict_masks(controllers, prompt, state)

    def evaluate(mask, detailed=False):
        return _candidate(pipeline, entry, base, pixels_base, base_frames, group, mask, device,
                          identity, full_score, detailed=detailed, dino=dino)

    initial = evaluate([0.5] * 30)
    oracle = optimize_mask(evaluate, oracle_plan, group, 1, initial)
    masks.update({"full": [0.0] * 30, "global_.5": [0.5] * 30,
                  "current_only": [1.0] * 30, "oracle": oracle["mask"]})
    records, unit_dir = [], Path(video_root) / group_id
    unit_dir.mkdir(parents=True, exist_ok=False)
    for condition in FINAL_CONDITIONS:
        result = evaluate(masks[condition], detailed=True)
        if condition == "full" and not result["drift"]["exact"]:
            raise AssertionError(f"{group_id}: full P0 is not exact base")
        record = {"protocol": PROTOCOL, "prompt_id": group["prompt_id"], "seed": seed,
                  "target_chunk": 4, "condition": condition, "layer_release": masks[condition],
                  "checkpoint_sha256": entry["sha256"], "feature_sha256": feature_digest(prompt, state),
                  "editability": result["editability"], "D_drift": result["D_drift"],
                  "drift": result["drift"], "responsiveness": result["responsiveness"],
                  "rng": result["rng"], "history_partition": result["history_partition"],
                  "chunk_latent_sha256": result["chunk_latent_sha256"],
                  "preservation": {"outside_exact": True,
                                   "outside_max_abs_after_vae": result["outside_max_abs_after_vae"],
                                   "identity": result["appearance"]},
                  "boundary": result["boundary"], "cost": {"status": "invalid_diagnostic"}}
        if condition == "oracle":
            record["optimization"] = {"lambda": 0.2, "best_loss": oracle["best_loss"],
                                      "trace": oracle["trace"]}
        for name, frames in result["pixels"].items():
            emedia.write_video(unit_dir / f"{condition}_{name}.mp4",
                               emedia.frames_to_uint8(frames[0]),
                               ex.generation_geometry(config)["fps"])
        records.append(record)
        print(f"[heldout] {group_id} {condition} R={record['editability']['R']} "
              f"D={record['D_drift']:.6g}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/edit_ready_mvp.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "validation/mask_distillation_heldout_manifest.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mask_distillation_plan.json")
    parser.add_argument("--oracle-plan", type=Path, default=ROOT / "configs/oracle_layer_mask_plan.json")
    parser.add_argument("--controllers", type=Path, default=ROOT / "validation/mask_controllers_m1a.pt")
    parser.add_argument("--teacher-dataset", type=Path,
                        default=ROOT / "validation/mask_teacher_dataset_m1a.json")
    parser.add_argument("--training-report", type=Path,
                        default=ROOT / "validation/mask_controller_training_m1a.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True); parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reviewed", action="store_true"); parser.add_argument("--heldout-approved", action="store_true")
    args = parser.parse_args()
    if not args.reviewed or not args.heldout_approved:
        parser.error("M1-A held-out evaluation requires explicit approval flags")
    config, manifest = ex.read_config(args.config), json.loads(args.manifest.read_text())
    plan, oracle_plan = json.loads(args.plan.read_text()), json.loads(args.oracle_plan.read_text())
    if plan.get("status") != "frozen_before_heldout" or plan.get("protocol") != PROTOCOL:
        parser.error("M1-A artifacts must be hash-locked before held-out evaluation")
    locked = plan["heldout"]
    observed_artifacts = (ec.sha256_file(args.controllers), ec.sha256_file(args.teacher_dataset),
                          ec.sha256_file(args.training_report))
    expected_artifacts = (locked["controller_checkpoint_sha256"], locked["teacher_dataset_sha256"],
                          locked["training_report_sha256"])
    if (ec.sha256_file(args.manifest) != locked["manifest_sha256"] or
            observed_artifacts != expected_artifacts):
        parser.error("held-out manifest or locked training artifact hash mismatch")
    groups = groups_from_manifest(config, manifest, "mask_distillation_heldout_m1a", HELDOUT_SEEDS)
    groups = ex.shard(groups, args.shard, args.shards)
    args.cache_dir.mkdir(parents=True, exist_ok=False); args.video_root.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = ex.device_for(args.gpu)
    kernel_checks = cuda_kernel_invariants(device)
    geometry = validate_model_geometry(config, {"history_partition": {
        "required_sink_size": 3, "required_num_frame_per_block": 3,
        "required_local_attention_frames": 12}})
    identity = ex.run_provenance(config, args.config, extra={
        "protocol": PROTOCOL, "stage": "M1A_heldout_controller_evaluation",
        "plan_sha256": ec.sha256_file(args.plan), "manifest_sha256": ec.sha256_file(args.manifest),
        "controller_sha256": ec.sha256_file(args.controllers), "kernel_invariants": kernel_checks,
        "teacher_dataset_sha256": ec.sha256_file(args.teacher_dataset),
        "training_report_sha256": ec.sha256_file(args.training_report),
        "history_geometry": geometry, "shard": args.shard, "shards": args.shards})
    pipeline, controllers = load_pipeline(config, device), load_controllers(args.controllers, "cpu")
    if any(parameter.requires_grad for parameter in pipeline.generator.parameters()):
        raise AssertionError("LongLive must remain frozen")
    if controllers["metadata"]["dataset_sha256"] != locked["teacher_dataset_sha256"]:
        raise AssertionError("controller metadata teacher hash mismatch")
    dino = em.DinoFeatureDistance(ROOT / config["model"]["dino"], device=device)
    records = []
    try:
        for group in groups:
            records.extend(run_unit(pipeline, config, group, device, args.cache_dir,
                                    args.video_root, identity, plan, oracle_plan, controllers, dino))
    except Exception as error:
        ex.write_json(args.output, {"experiment": "mask_distillation_heldout_m1a", "status": "failed",
                                   "provenance": identity, "cases": records,
                                   "error": f"{type(error).__name__}: {error}"})
        raise
    ex.write_json(args.output, {"experiment": "mask_distillation_heldout_m1a", "schema": 1,
                               "status": "complete", "provenance": identity, "cases": records})


if __name__ == "__main__":
    main()
