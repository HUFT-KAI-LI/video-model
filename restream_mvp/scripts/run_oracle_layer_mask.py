#!/usr/bin/env python3
"""Optimize per-case 30-layer history-release masks with frozen exact metrics."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code/LongLive")]

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_media as emedia  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.history_gate import history_gate, history_layer_release  # noqa: E402
from restream.oracle_layer_mask import (  # noqa: E402
    BASELINES, PROTOCOL, manifest_groups, oracle_name, validate_plan)
from restream.runtime import load_pipeline  # noqa: E402
from scripts.run_chunk_edit import _dino_boundary, _load_entry, _rng_check, tensor_digest  # noqa: E402
from scripts.run_history_component_screen import paired_editability, validate_model_geometry  # noqa: E402


def cuda_kernel_invariants(device):
    spec = importlib.util.spec_from_file_location(
        "oracle_attention", ROOT / "code/LongLive/wan/modules/attention.py")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    q = torch.randn(1, 6, 2, 64, device=device, dtype=torch.bfloat16)
    kh = torch.randn(1, 9, 2, 64, device=device, dtype=torch.bfloat16)
    vh = torch.randn_like(kh)
    kc = torch.randn(1, 6, 2, 64, device=device, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    full = kernel.attention(q, torch.cat([kh, kc], 1), torch.cat([vh, vc], 1))
    current = kernel.attention(q, kc, vc)
    zero = kernel.layer_release_attention(q, kh, vh, kc, vc, 0)
    one = kernel.layer_release_attention(q, kh, vh, kc, vc, 1)
    half = kernel.layer_release_attention(q, kh, vh, kc, vc, .5)
    stage_c = kernel.gated_attention(q, kh, vh, kc, vc, .5)
    if not torch.equal(zero, full):
        raise AssertionError("release=0 is not bit-exact native full attention")
    if not torch.equal(one, current):
        raise AssertionError("release=1 is not bit-exact native current-only attention")
    if not torch.equal(half, stage_c):
        raise AssertionError("release=.5 is not bit-exact Stage-C global .5")
    return {"release_0_native_full_exact": True, "release_1_native_current_exact": True,
            "release_half_stage_c_exact": True, "dtype": str(q.dtype)}


def _layer_snapshot(pipeline, block):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    observed = [getattr(m, "last_history_layer_tokens", None) for m in modules]
    if len(modules) != 30 or any(value is None for value in observed):
        raise AssertionError("all 30 attention layers must report layer-release routing")
    signatures = {tuple(value[name] for name in ("history", "current")) for value in observed}
    if len(signatures) != 1:
        raise AssertionError("layer-release token partitions differ across layers")
    tokens = dict(observed[0])
    per_frame = tokens["current"] // block
    return {"tokens": tokens,
            "latent_frames": {name: value // per_frame for name, value in tokens.items()},
            "tokens_per_latent_frame": per_frame, "modules_checked": len(modules)}


def _candidate(pipeline, checkpoint_entry, base, pixels_base, base_frames, group, mask,
               device, identity, full_score, *, context="layer", detailed=False, dino=None):
    target, block = group["target_chunk"], int(pipeline.num_frame_per_block)
    span = er.chunk_frame_slice(target, block, pixels_base.shape[1])
    sl = slice(span["pixel_start"], span["pixel_end"])
    results, pixels, rng, partitions = {}, {}, {}, {}
    for name, prompt, binding in (("replay", group["base_prompt"], "restore"),
                                   ("text_rebind", group["edit_prompt"], "reset")):
        checkpoint = _load_entry(checkpoint_entry, device, identity["model_checkpoint_sha256"],
                                 identity["config_hash"])
        manager = (history_layer_release(pipeline, mask) if context == "layer"
                   else history_gate(pipeline, float(mask)))
        with manager:
            replay = er.replay_chunk(
                pipeline, checkpoint, prompt, device=device, noise="from-cache",
                noise_source="stored", crossattn=binding, restore_rng=True,
                expected_model_hash=identity["model_checkpoint_sha256"],
                expected_config_hash=identity["config_hash"], time_it=False)
            if context == "layer":
                partitions[name] = _layer_snapshot(pipeline, block)
        rng[name] = _rng_check(checkpoint, replay.recorded_noise)
        if not rng[name].get("exact"):
            raise AssertionError(f"{group['prompt_id']}: {name} replay RNG mismatch")
        results[name] = replay.latents
        assembled = er.assemble_latents(base.latents, replay.latents, target, block)
        er.outside_exact(assembled, base.latents, target, block)
        pixels[name] = er.decode_latents(pipeline, assembled)
    responsiveness = {
        name: em.evaluate_probe(group["probe"], base_frames, pixels[name][0, sl])
        for name in results}
    responsiveness["full_regeneration"] = {"S_proxy": full_score}
    editability = paired_editability(responsiveness)
    base_chunk = base.latents[:, target * block:(target + 1) * block]
    drift = em.latent_stats(base_chunk, results["replay"])
    output = {"editability": editability, "D_drift": drift["mse"], "drift": drift,
              "responsiveness": responsiveness, "rng": rng,
              "chunk_latent_sha256": {name: tensor_digest(value) for name, value in results.items()},
              "history_partition": partitions or None}
    if detailed:
        boundaries, outside_pixels = {}, {}
        base_boundary = em.boundary_latent(base.latents, target, block)
        base_boundary_px = em.boundary_pixels(pixels_base, target, block)
        appearance = {"status": "not_measured"}
        if dino is not None:
            appearance = {"status": "measured", "metric": "DINOv2 appearance proxy",
                          "model": dino.identity}
        for name in results:
            assembled = er.assemble_latents(base.latents, results[name], target, block)
            after = em.boundary_latent(assembled, target, block)
            after_px = em.boundary_pixels(pixels[name], target, block)
            boundaries[name] = {
                "latent": {"base": base_boundary, "edited": after,
                           "delta": em.boundary_delta(after, base_boundary)},
                "pixel": {"base": base_boundary_px, "edited": after_px,
                          "delta": em.boundary_delta(after_px, base_boundary_px)}}
            outside_pixels[name] = max(
                float((pixels_base[:, :sl.start] - pixels[name][:, :sl.start]).abs().max())
                if sl.start else 0.0,
                float((pixels_base[:, sl.stop:] - pixels[name][:, sl.stop:]).abs().max())
                if sl.stop < pixels_base.shape[1] else 0.0)
            if dino is not None:
                appearance[name] = dino.distance(base_frames, pixels[name][0, sl])
                boundaries[name]["dino"] = _dino_boundary(dino, pixels_base, pixels[name], span)
        output.update({"boundary": boundaries, "outside_max_abs_after_vae": outside_pixels,
                       "appearance": appearance, "pixels": pixels})
    return output


def _loss(candidate, mask, drift_lambda, gamma_l1):
    return (-candidate["editability"]["E"] + drift_lambda * candidate["D_drift"]
            + gamma_l1 * sum(mask))


def optimize_mask(evaluate, plan, group, lambda_index, initial):
    cfg = plan["optimization"]
    drift_lambda = float(cfg["lambdas"][lambda_index])
    gamma_l1 = float(cfg["gamma_l1"])
    z = torch.zeros(cfg["layer_count"], dtype=torch.float64)
    best_mask = [0.5] * cfg["layer_count"]
    best_loss = _loss(initial, best_mask, drift_lambda, gamma_l1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(group["seed"] * 1009 + group["prompt_index"] * 97 + lambda_index)
    trace = [{"iteration": -1, "side": "initial", "loss": best_loss,
              "E": initial["editability"]["E"], "D_drift": initial["D_drift"]}]
    for iteration in range(int(cfg["iterations"])):
        a_t = cfg["gain_a"] / ((iteration + 1 + cfg["stability_A"]) ** cfg["alpha"])
        c_t = cfg["perturbation_c"] / ((iteration + 1) ** cfg["gamma"])
        delta = torch.randint(0, 2, z.shape, generator=generator, dtype=torch.int64).double() * 2 - 1
        observations = []
        for side in (1.0, -1.0):
            candidate_z = (z + side * c_t * delta).clamp(-cfg["logit_clip"], cfg["logit_clip"])
            mask = torch.sigmoid(candidate_z).tolist()
            result = evaluate(mask)
            value = _loss(result, mask, drift_lambda, gamma_l1)
            observations.append(value)
            trace.append({"iteration": iteration, "side": "plus" if side > 0 else "minus",
                          "loss": value, "E": result["editability"]["E"],
                          "D_drift": result["D_drift"], "mask": mask})
            if value < best_loss:
                best_loss, best_mask = value, mask
        gradient = ((observations[0] - observations[1]) / (2 * c_t)) * delta
        z = (z - a_t * gradient).clamp(-cfg["logit_clip"], cfg["logit_clip"])
    return {"condition": oracle_name(drift_lambda), "lambda": drift_lambda,
            "best_loss": best_loss, "mask": best_mask, "trace": trace}


def run_unit(pipeline, config, group, device, output_dir, cache_dir, identity, plan, dino):
    block, target, seed = int(pipeline.num_frame_per_block), 4, group["seed"]
    generation_args = dict(seed=seed, model_hash=identity["model_checkpoint_sha256"],
                           model_record=identity["model_identity"],
                           config_digest=identity["config_hash"], noise_source="global")
    group_id = f"oracle_{group['prompt_id']}_seed{seed}"
    with history_gate(pipeline, 1.0):
        base = er.stream_generate(pipeline, ex.sample_noise(config, device, seed),
                                  group["base_prompt"], sample_id=group_id,
                                  cache_dir=cache_dir, save_chunks=[target], **generation_args)
        regenerated = er.stream_generate(pipeline, ex.sample_noise(config, device, seed),
                                         group["edit_prompt"], sample_id=group_id + "_fullregen",
                                         save_chunks=[], **generation_args)
    pixels_base = er.decode_latents(pipeline, base.latents)
    pixels_full = er.decode_latents(pipeline, regenerated.latents)
    span = er.chunk_frame_slice(target, block, pixels_base.shape[1])
    sl = slice(span["pixel_start"], span["pixel_end"])
    base_frames = pixels_base[0, sl]
    full_score = em.evaluate_probe(group["probe"], base_frames, pixels_full[0, sl])["S_proxy"]
    entry = base.checkpoint_entries[target]

    def evaluate(mask, context="layer", detailed=False):
        return _candidate(pipeline, entry, base, pixels_base, base_frames, group, mask, device,
                          identity, full_score, context=context, detailed=detailed, dino=dino)

    initial = evaluate([0.5] * 30)
    stage_c = evaluate(0.5, context="global")
    if initial["chunk_latent_sha256"] != stage_c["chunk_latent_sha256"]:
        raise AssertionError(f"{group_id}: all-half layer mask differs from Stage-C global .5")
    optimized = [optimize_mask(evaluate, plan, group, index, initial)
                 for index in range(len(plan["optimization"]["lambdas"]))]
    final_specs = [("full", [0.0] * 30), ("global_.5", [0.5] * 30),
                   ("current_only", [1.0] * 30)]
    final_specs += [(item["condition"], item["mask"]) for item in optimized]
    records = []
    unit_dir = Path(output_dir) / group_id
    unit_dir.mkdir(parents=True, exist_ok=False)
    for condition, mask in final_specs:
        result = evaluate(mask, detailed=True)
        if condition == "full" and not result["drift"]["exact"]:
            raise AssertionError(f"{group_id}: zero release P0 is not exact base")
        record = {"protocol": PROTOCOL, "prompt_id": group["prompt_id"], "seed": seed,
                  "target_chunk": target, "condition": condition, "layer_release": mask,
                  "checkpoint_sha256": entry["sha256"], "editability": result["editability"],
                  "D_drift": result["D_drift"], "drift": result["drift"],
                  "responsiveness": result["responsiveness"], "rng": result["rng"],
                  "chunk_latent_sha256": result["chunk_latent_sha256"],
                  "history_partition": result["history_partition"],
                  "preservation": {"outside_exact": True,
                                   "outside_max_abs_after_vae": result["outside_max_abs_after_vae"],
                                   "identity": result["appearance"]},
                  "boundary": result["boundary"], "cost": {"status": "invalid_diagnostic"}}
        if condition.startswith("oracle_"):
            item = next(value for value in optimized if value["condition"] == condition)
            record["optimization"] = {key: value for key, value in item.items() if key != "condition"}
        for name, frames in result["pixels"].items():
            emedia.write_video(unit_dir / f"{condition}_{name}.mp4",
                               emedia.frames_to_uint8(frames[0]),
                               ex.generation_geometry(config)["fps"])
        records.append(record)
        print(f"[oracle] {group_id} {condition} E={record['editability']['E']:+.6f} "
              f"R={record['editability']['R']} D={record['D_drift']:.6g}", flush=True)
    by_condition = {record["condition"]: record for record in records}
    if all(by_condition["current_only"]["chunk_latent_sha256"][name] ==
           by_condition["full"]["chunk_latent_sha256"][name]
           for name in ("replay", "text_rebind")):
        raise AssertionError(f"{group_id}: current-only intervention is inert")
    return records, {"prompt_id": group["prompt_id"], "seed": seed,
                     "checkpoint_sha256": entry["sha256"],
                     "all_half_equals_stage_c": True,
                     "optimizations": [{key: value for key, value in item.items() if key != "trace"}
                                       for item in optimized]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/edit_ready_mvp.yaml")
    parser.add_argument("--manifest", type=Path, default=ROOT / "validation/oracle_layer_mask_manifest.json")
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/oracle_layer_mask_plan.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--oracle-approved", action="store_true")
    args = parser.parse_args()
    if not args.reviewed or not args.oracle_approved:
        parser.error("oracle GPU optimization requires --reviewed and --oracle-approved")
    config, manifest, plan = ex.read_config(args.config), json.loads(args.manifest.read_text()), json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_m0m1_oracle" or plan.get("protocol") != PROTOCOL:
        parser.error("oracle plan must be frozen and protocol matched")
    if ec.sha256_file(args.manifest) != plan["manifest"]["sha256"]:
        parser.error("oracle manifest differs from frozen plan")
    groups = manifest_groups(config, manifest)
    validate_plan(groups, plan)
    groups = ex.shard(groups, args.shard, args.shards)
    root = ROOT / "validation/oracle_layer_masks"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"oracle_shard{args.shard}_", dir=root))
    output, video_root, cache_dir = (args.output or run_dir / "results.json",
                                     args.video_root or run_dir / "videos",
                                     args.cache_dir or run_dir / "cache")
    for path in (video_root, cache_dir):
        path.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = ex.device_for(args.gpu)
    if device.type != "cuda":
        parser.error("oracle runner requires CUDA")
    kernel_checks = cuda_kernel_invariants(device)
    geometry = validate_model_geometry(config, {"history_partition": {
        "required_sink_size": 3, "required_num_frame_per_block": 3,
        "required_local_attention_frames": 12}})
    identity = ex.run_provenance(config, args.config, extra={
        "protocol": PROTOCOL, "stage": plan["stage"], "plan_sha256": ec.sha256_file(args.plan),
        "manifest_sha256": ec.sha256_file(args.manifest), "optimization": plan["optimization"],
        "history_geometry": geometry, "shard": args.shard, "shards": args.shards,
        "kernel_invariants": kernel_checks,
        "timing_status": "invalid_diagnostic"})
    pipeline = load_pipeline(config, device)
    if any(parameter.requires_grad for parameter in pipeline.generator.parameters()):
        raise AssertionError("LongLive must remain fully frozen")
    dino = em.DinoFeatureDistance(ROOT / config["model"]["dino"], device=device)
    records, units = [], []
    try:
        for group in groups:
            unit_records, unit = run_unit(pipeline, config, group, device, video_root,
                                          cache_dir, identity, plan, dino)
            records.extend(unit_records)
            units.append(unit)
    except Exception as error:
        ex.write_json(output, {"experiment": "oracle_layer_release_m0m1", "status": "failed",
                               "provenance": identity, "cases": records, "units": units,
                               "error": f"{type(error).__name__}: {error}"})
        raise
    ex.write_json(output, {"experiment": "oracle_layer_release_m0m1", "schema": 1,
                           "status": "complete", "provenance": identity, "config": config,
                           "cases": records, "units": units})
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
