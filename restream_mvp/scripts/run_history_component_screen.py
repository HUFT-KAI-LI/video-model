#!/usr/bin/env python3
"""D1 fixed-history sink/old/recent component screen."""
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
from restream import edit_media as emedia  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.history_components import (  # noqa: E402
    CONDITIONS, PROTOCOL, condition_spec, manifest_groups, validate_screen)
from restream.history_gate import (  # noqa: E402
    history_component_gates, history_gate, history_path_gates)
from restream.runtime import load_pipeline  # noqa: E402
from scripts.run_chunk_edit import _dino_boundary, _load_entry, _rng_check, tensor_digest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def paired_editability(response):
    p0 = response["replay"]["S_proxy"]
    p1 = response["text_rebind"]["S_proxy"]
    full = response["full_regeneration"]["S_proxy"]
    effect = p1 - p0
    return {"S_P0": p0, "S_P1": p1, "S_full": full, "E": effect,
            "R": effect / full if full > 0 else None,
            "ratio_status": "valid" if full > 0 else "nonpositive_full_reference"}


def _condition_context(pipeline, condition, spec_fn=condition_spec):
    spec = spec_fn(condition)
    if spec["kind"] == "global":
        return history_gate(pipeline, spec["gate"])
    if spec["kind"] == "path":
        return history_path_gates(pipeline, spec["score_gate"], spec["value_gate"])
    return history_component_gates(pipeline, spec["gates"])


def _condition_record(condition, spec_fn=condition_spec):
    spec = spec_fn(condition)
    mechanism = {"global": "stage_c_global", "components": "component",
                 "path": "attention_path"}[spec["kind"]]
    return {"condition": condition, "mechanism": mechanism,
            "history_gate": spec.get("gate"),
            "history_component_gates": spec.get("gates"),
            "history_path_gates": ({"score": spec["score_gate"], "value": spec["value_gate"]}
                                   if spec["kind"] == "path" else None)}


def _component_partition_snapshot(pipeline, block):
    root = getattr(pipeline, "generator", pipeline)
    modules = [module for module in root.modules()
               if module.__class__.__name__ == "CausalWanSelfAttention"]
    observed = [getattr(module, "last_history_component_tokens", None) for module in modules]
    if not observed or any(value is None for value in observed):
        raise AssertionError("component partition was not recorded by every attention module")
    signatures = {tuple(value[name] for name in ("sink", "old", "recent", "current"))
                  for value in observed}
    if len(signatures) != 1:
        raise AssertionError("attention modules observed different history partitions")
    tokens = dict(observed[0])
    if tokens["current"] % int(block):
        raise AssertionError("current query tokens do not align to AR latent frames")
    tokens_per_frame = tokens["current"] // int(block)
    if tokens_per_frame <= 0 or any(value % tokens_per_frame for value in tokens.values()):
        raise AssertionError("history components do not align to latent-frame boundaries")
    return {"tokens": tokens,
            "latent_frames": {name: value // tokens_per_frame for name, value in tokens.items()},
            "tokens_per_latent_frame": tokens_per_frame,
            "modules_checked": len(modules)}


def _path_partition_snapshot(pipeline, block):
    root = getattr(pipeline, "generator", pipeline)
    modules = [module for module in root.modules()
               if module.__class__.__name__ == "CausalWanSelfAttention"]
    observed = [getattr(module, "last_history_path_tokens", None) for module in modules]
    if not observed or any(value is None for value in observed):
        raise AssertionError("path token partition was not recorded by every attention module")
    signatures = {tuple(value[name] for name in ("history", "current")) for value in observed}
    if len(signatures) != 1:
        raise AssertionError("attention modules observed different path partitions")
    tokens = dict(observed[0])
    if tokens["current"] % int(block):
        raise AssertionError("current path tokens do not align to AR latent frames")
    tokens_per_frame = tokens["current"] // int(block)
    if tokens_per_frame <= 0 or any(value % tokens_per_frame for value in tokens.values()):
        raise AssertionError("path tokens do not align to latent-frame boundaries")
    return {"tokens": tokens,
            "latent_frames": {name: value // tokens_per_frame for name, value in tokens.items()},
            "tokens_per_latent_frame": tokens_per_frame, "modules_checked": len(modules)}


def validate_model_geometry(config, plan):
    upstream = ex.read_config(ROOT / config["model"]["upstream_config"])
    partition = plan["history_partition"]
    observed = {
        "sink_size": int(upstream["model_kwargs"]["sink_size"]),
        "num_frame_per_block": int(upstream["num_frame_per_block"]),
        "local_attn_size": int(upstream["model_kwargs"]["local_attn_size"]),
    }
    expected = {
        "sink_size": int(partition["required_sink_size"]),
        "num_frame_per_block": int(partition["required_num_frame_per_block"]),
        "local_attn_size": int(partition["required_local_attention_frames"]),
    }
    if observed != expected:
        raise ValueError(f"D1 history geometry mismatch: observed={observed}, expected={expected}")
    if observed["sink_size"] != observed["num_frame_per_block"]:
        raise ValueError("D1 requires exactly one AR chunk in the sink")
    return observed


def run_group(pipeline, config, group, device, output_dir, cache_dir, identity, dino=None):
    """Generate full-history references once and replay every D1 intervention."""
    with history_gate(pipeline, 1.0):
        return _run_fixed_group(pipeline, config, group, device, output_dir,
                                cache_dir, identity, dino)


def _run_fixed_group(pipeline, config, group, device, output_dir, cache_dir, identity, dino,
                     *, protocol=PROTOCOL, spec_fn=condition_spec,
                     full_condition="full_history", stage="D1"):
    block = int(pipeline.num_frame_per_block)
    seed = int(group["seed"])
    targets = group["targets"]
    base_prompt, edit_prompt = group["base_prompt"], group["edit_prompt"]
    noise_source = config["generation"].get("noise_source", "global")
    noise = ex.sample_noise(config, device, seed)
    num_chunks = noise.shape[1] // block
    ex.target_chunks(config, targets, num_chunks)
    group_id = f"edit_{group['prompt_id']}_seed{seed}"
    generation_args = dict(seed=seed, model_hash=identity["model_checkpoint_sha256"],
                           model_record=identity["model_identity"],
                           config_digest=identity["config_hash"], noise_source=noise_source)
    base = er.stream_generate(pipeline, ex.sample_noise(config, device, seed), base_prompt,
                              sample_id=group_id, cache_dir=cache_dir,
                              save_chunks=targets, **generation_args)
    regenerated = er.stream_generate(
        pipeline, ex.sample_noise(config, device, seed), edit_prompt,
        sample_id=f"{group_id}_fullregen", cache_dir=None, save_chunks=[], **generation_args)
    pixels_base = er.decode_latents(pipeline, base.latents)
    pixels_full = er.decode_latents(pipeline, regenerated.latents)
    fps = ex.generation_geometry(config)["fps"]
    records = []
    for target in targets:
        entry = base.checkpoint_entries[target]
        base_chunk = base.latents[:, target * block:(target + 1) * block]
        full_chunk = regenerated.latents[:, target * block:(target + 1) * block]
        span = er.chunk_frame_slice(target, block, pixels_base.shape[1])
        sl = slice(span["pixel_start"], span["pixel_end"])
        base_frames, full_frames = pixels_base[0, sl], pixels_full[0, sl]
        for condition in group["conditions_by_target"][target]:
            case_id = f"{group_id}_chunk{target}_{condition}"
            case_dir = Path(output_dir) / case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            results, assembled, pixels, rng, partitions = {}, {}, {}, {}, {}
            for name, prompt, binding in (("replay", base_prompt, "restore"),
                                           ("text_rebind", edit_prompt, "reset")):
                checkpoint = _load_entry(entry, device, identity["model_checkpoint_sha256"],
                                         identity["config_hash"])
                with _condition_context(pipeline, condition, spec_fn):
                    result = er.replay_chunk(
                        pipeline, checkpoint, prompt, device=device, noise="from-cache",
                        noise_source=noise_source, crossattn=binding, restore_rng=True,
                        expected_model_hash=identity["model_checkpoint_sha256"],
                        expected_config_hash=identity["config_hash"])
                    kind = spec_fn(condition)["kind"]
                    if kind == "components":
                        partitions[name] = _component_partition_snapshot(pipeline, block)
                    elif kind == "path":
                        partitions[name] = _path_partition_snapshot(pipeline, block)
                results[name] = result.latents
                rng[name] = _rng_check(checkpoint, result.recorded_noise)
                if rng[name].get("comparable") and not rng[name]["exact"]:
                    raise AssertionError(f"{case_id}: {name} replay noise differs from checkpoint")
                assembled[name] = er.assemble_latents(base.latents, result.latents, target, block)
                er.outside_exact(assembled[name], base.latents, target, block)
                pixels[name] = er.decode_latents(pipeline, assembled[name])
            drift = em.latent_stats(base_chunk, results["replay"])
            if condition == full_condition and not drift["exact"]:
                raise AssertionError(f"{case_id}: full-history P0 must be exact base")
            digests = {name: tensor_digest(value) for name, value in results.items()}
            digests.update(original=tensor_digest(base_chunk),
                           full_regeneration=tensor_digest(full_chunk))
            responsiveness = {
                name: em.evaluate_probe(group["probe"], base_frames, frames)
                for name, frames in (("replay", pixels["replay"][0, sl]),
                                     ("text_rebind", pixels["text_rebind"][0, sl]),
                                     ("full_regeneration", full_frames))}
            scores = paired_editability(responsiveness)
            boundary = {}
            for name in results:
                before = em.boundary_latent(base.latents, target, block)
                after = em.boundary_latent(assembled[name], target, block)
                before_px = em.boundary_pixels(pixels_base, target, block)
                after_px = em.boundary_pixels(pixels[name], target, block)
                boundary[name] = {
                    "latent": {"base": before, "edited": after,
                               "delta": em.boundary_delta(after, before)},
                    "pixel": {"base": before_px, "edited": after_px,
                              "delta": em.boundary_delta(after_px, before_px)}}
            appearance = {"status": "not_measured", "metric": "DINOv2 appearance proxy",
                          "note": "Appearance distance is not a validated identity score."}
            if dino is not None:
                appearance = {"status": "measured", "metric": "DINOv2 appearance proxy",
                              "model": dino.identity,
                              **{name: dino.distance(base_frames, pixels[name][0, sl])
                                 for name in results}}
                for name in results:
                    boundary[name]["dino"] = _dino_boundary(
                        dino, pixels_base, pixels[name], span)
            outside_pixels = {}
            for name in results:
                outside_pixels[name] = max(
                    float((pixels_base[:, :sl.start] - pixels[name][:, :sl.start]).abs().max())
                    if sl.start else 0.0,
                    float((pixels_base[:, sl.stop:] - pixels[name][:, sl.stop:]).abs().max())
                    if sl.stop < pixels_base.shape[1] else 0.0)
            record = {
                "protocol": protocol, "sample_id": case_id,
                "prompt_id": group["prompt_id"], "seed": seed, "target_chunk": target,
                **_condition_record(condition, spec_fn),
                "policies": {"replay": "P0", "text_rebind": "P1"},
                "evidence": group.get("evidence", "directional"), "num_chunks": num_chunks,
                "base_prompt": base_prompt, "edit_prompt": edit_prompt,
                "noise_sha256": tensor_digest(noise), "noise_source": noise_source,
                "chunk_latent_sha256": digests, "rng": rng,
                "history_partition": partitions or None,
                "responsiveness": responsiveness, "editability": scores,
                "D_drift": drift["mse"], "D_drift_metric": "target_chunk_latent_mse",
                "drift": {"latent": drift,
                          "pixel": em.pixel_stats(base_frames, pixels["replay"][0, sl])},
                "sanity": {"full_history_P0_exact_base":
                           drift["exact"] if condition == full_condition else None,
                           "fixed_references_unmodified_full_history": True},
                "preservation": {"outside_exact": True, "policies_checked": ["P0", "P1"],
                                 "chunks_checked": num_chunks - 1,
                                 "outside_max_abs_after_vae": outside_pixels,
                                 "identity": appearance},
                "boundary": boundary,
                "cache": {key: value for key, value in entry.items() if key != "checkpoint"},
                "cost": {"status": "invalid_diagnostic", "passed": None,
                         "reason": f"{stage} mechanism screen includes incomparable attention operators."},
            }
            videos = {"original": pixels_base, "full_regeneration": pixels_full, **pixels}
            for name, tensor in videos.items():
                emedia.write_video(case_dir / f"{name}.mp4", emedia.frames_to_uint8(tensor[0]), fps)
            labels = {"original": "B0 full history", "full_regeneration": "FullReg P1",
                      "replay": f"P0 {condition}", "text_rebind": f"P1 {condition}"}
            emedia.write_comparison(case_dir / "comparison.mp4", videos, fps, labels=labels)
            record["videos"] = {name: str(case_dir / f"{name}.mp4") for name in videos}
            record["videos"]["comparison"] = str(case_dir / "comparison.mp4")
            ex.write_json(case_dir / "metrics.json", record)
            records.append(record)
            print(f"[pair] {case_id} E={scores['E']:+.6f} R={scores['R']} "
                  f"D_drift={drift['mse']:.6g}", flush=True)
    return records


def assert_component_invariants(records):
    by_unit = {}
    for record in records:
        key = (record["prompt_id"], record["seed"], record["target_chunk"])
        if record["condition"] in by_unit.setdefault(key, {}):
            raise AssertionError(f"{key}: duplicate condition {record['condition']}")
        by_unit[key][record["condition"]] = record
    for key, cases in by_unit.items():
        if set(cases) != set(CONDITIONS):
            raise AssertionError(f"{key}: incomplete D1 condition set")
        variants = ("replay", "text_rebind")
        if key[2] == 1:
            for variant in variants:
                if (cases["global_release"]["chunk_latent_sha256"][variant] !=
                        cases["sink_release"]["chunk_latent_sha256"][variant]):
                    raise AssertionError(f"{key}: chunk1 global and sink differ for {variant}")
                full = cases["full_history"]["chunk_latent_sha256"][variant]
                for condition in ("old_release", "recent_release", "non_sink_release"):
                    if cases[condition]["chunk_latent_sha256"][variant] != full:
                        raise AssertionError(
                            f"{key}: chunk1 empty {condition} differs from full for {variant}")
            expected_partition = {"sink": 3, "old": 0, "recent": 0, "current": 3}
        if key[2] == 4:
            full = cases["full_history"]["chunk_latent_sha256"]
            for condition in CONDITIONS:
                if condition == "full_history":
                    continue
                if all(cases[condition]["chunk_latent_sha256"][variant] == full[variant]
                       for variant in variants):
                    raise AssertionError(f"{key}: {condition} is inert at chunk4")
            expected_partition = {"sink": 3, "old": 3, "recent": 3, "current": 3}
        for condition, case in cases.items():
            if condition == "global_release":
                if case.get("history_partition") is not None:
                    raise AssertionError(f"{key}: global baseline used the component router")
                continue
            partitions = case.get("history_partition", {})
            if set(partitions) != set(variants):
                raise AssertionError(f"{key}: missing component partition audit for {condition}")
            if any(partitions[variant]["latent_frames"] != expected_partition
                   for variant in variants):
                raise AssertionError(f"{key}: unexpected component partition for {condition}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/edit_ready_mvp.yaml")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "validation/history_component_screen_manifest.json")
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "configs/history_component_screen_plan.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dino", action="store_true",
                        help="Required frozen DINOv2 appearance diagnostic")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--screen-approved", action="store_true")
    args = parser.parse_args()
    if not args.reviewed or not args.screen_approved:
        parser.error("D1 GPU generation requires --reviewed and --screen-approved")
    if not args.dino:
        parser.error("D1 plan requires --dino for the appearance-preservation diagnostic")
    config = ex.read_config(args.config)
    manifest = json.loads(args.manifest.read_text())
    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_d1_screen" or plan.get("protocol") != PROTOCOL:
        parser.error("D1 screen plan must be frozen and protocol-matched")
    if ec.sha256_file(args.manifest) != plan["manifest"]["sha256"]:
        parser.error("D1 manifest differs from the frozen screen plan")
    try:
        geometry = validate_model_geometry(config, plan)
        all_groups = manifest_groups(config, manifest)
        validate_screen(all_groups, plan)
        groups = ex.shard(all_groups, args.shard, args.shards)
    except ValueError as error:
        parser.error(str(error))
    if not groups:
        parser.error("No D1 groups selected")
    root = ROOT / "validation/history_components"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"d1_shard{args.shard}_", dir=root))
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
        parser.error("D1 runner requires working CUDA")
    torch.set_grad_enabled(False)
    identity = ex.run_provenance(config, args.config, extra={
        "protocol": PROTOCOL, "stage": "D1_sink_old_recent_decomposition",
        "manifest_path": str(args.manifest), "manifest_sha256": ec.sha256_file(args.manifest),
        "screen_plan_path": str(args.plan), "screen_plan_sha256": ec.sha256_file(args.plan),
        "screen_plan_status": plan["status"], "history_geometry": geometry,
        "conditions": plan["conditions"], "shard": args.shard, "shards": args.shards,
        "video_root": str(video_root), "cache_dir": str(cache_dir),
        "timing_status": "invalid_diagnostic"})
    pipeline = load_pipeline(config, device)
    dino = em.DinoFeatureDistance(ROOT / config["model"]["dino"], device=device) if args.dino else None
    records = []
    try:
        for group in groups:
            records.extend(run_group(pipeline, config, group, device, video_root,
                                     cache_dir, identity, dino=dino))
        assert_component_invariants(records)
    except Exception as error:
        ex.write_json(output, {"experiment": "history_component_screen_d1",
                              "status": "failed", "provenance": identity,
                              "cases": records, "error": f"{type(error).__name__}: {error}"})
        raise
    payload = {"experiment": "history_component_screen_d1", "schema": 1,
               "status": "complete", "provenance": identity, "config": config,
               "cases": records,
               "screen": {"passed_invariants": True, "pairs": len(records),
                          "chunk_replays": 2 * len(records),
                          "interpretation": "exploratory_no_significance_test"}}
    ex.write_json(output, payload)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
