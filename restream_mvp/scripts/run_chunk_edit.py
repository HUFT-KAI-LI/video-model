"""Experiment B -- prompt-changed local replay (Gates B/C/D).

For every prompt pair the base video and the full-regeneration reference are
computed once; then, for every target chunk:

* **B0 original**         - full generation with the original prompt ``P0``;
* **B1 full regeneration**- full generation from scratch with ``P1`` (edit-success
  reference, but it changes the whole video);
* **B2 cached replay**    - only chunk ``k`` reopened from ``S_{k-1}`` with ``P0``;
* **B3 cached local edit**- only chunk ``k`` reopened from ``S_{k-1}`` with ``P1``
  after clearing the text cross-attention cache (**the MVP method**);
* **control**             - ``P1`` supplied but the cached ``P0`` text K/V kept
  (``crossattn=restore``); expected to be a no-op, which isolates "prompt changed"
  from "text binding changed".

Outside preservation is asserted with ``torch.equal`` on the pre-decode latents
(Gate C).  Boundary continuity, responsiveness proxies and cost are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_media as emedia  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.runtime import load_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def _rng_check(checkpoint, recorded) -> dict:
    stored = checkpoint.denoise_noise
    if not stored or not recorded:
        return {"comparable": False}
    maximum = max(float((left.float().cpu() - right.float().cpu()).abs().max().item())
                  for left, right in zip(stored, recorded))
    return {"comparable": True, "steps": len(stored), "max_abs": maximum, "exact": maximum == 0.0}


def _timed_load(entry) -> float:
    started = time.perf_counter()
    ec.load_edit_checkpoint(entry["path"])
    return time.perf_counter() - started


def _dino_boundary(dino, reference_pixels, edited_pixels, span) -> dict:
    total = reference_pixels.shape[1]
    left, right = span["pixel_start"], span["pixel_end"]
    values = {}
    if left > 0:
        values["left_base"] = dino.distance(reference_pixels[0, left - 1:left],
                                            reference_pixels[0, left:left + 1])
        values["left_edited"] = dino.distance(edited_pixels[0, left - 1:left],
                                              edited_pixels[0, left:left + 1])
    if right < total:
        values["right_base"] = dino.distance(reference_pixels[0, right - 1:right],
                                             reference_pixels[0, right:right + 1])
        values["right_edited"] = dino.distance(edited_pixels[0, right - 1:right],
                                               edited_pixels[0, right:right + 1])
    return values


def run_group(pipeline, config, group, device, output_dir, cache_dir, identity, dino=None) -> list:
    block = int(pipeline.num_frame_per_block)
    seed = int(group["seed"])
    base_prompt, edit_prompt = group["base_prompt"], group["edit_prompt"]
    targets = [int(value) for value in group["targets"]]
    noise = ex.sample_noise(config, device, seed)
    num_chunks = noise.shape[1] // block
    for target in targets:
        if target >= num_chunks:
            raise ValueError(f"Target chunk {target} outside {num_chunks} chunks")
    noise_source = config["generation"].get("noise_source", "global")
    fps = ex.generation_geometry(config)["fps"]
    group_id = f"edit_{group['prompt_id']}_seed{seed}"

    # ---- B0: original generation, capturing the edit cache -----------------
    base_started = time.perf_counter()
    base = er.stream_generate(
        pipeline, noise, base_prompt, sample_id=group_id, seed=seed,
        model_hash=identity["model_checkpoint_sha256"], model_record=identity["model_identity"],
        config_digest=identity["config_hash"], cache_dir=cache_dir, save_chunks=targets,
        noise_source=noise_source)
    base_seconds = time.perf_counter() - base_started

    # ---- B1: full regeneration from scratch with the new prompt ------------
    regen_started = time.perf_counter()
    regenerated = er.stream_generate(
        pipeline, noise, edit_prompt, sample_id=f"{group_id}_fullregen", seed=seed,
        model_hash=identity["model_checkpoint_sha256"], model_record=identity["model_identity"],
        config_digest=identity["config_hash"], cache_dir=None, save_chunks=[],
        noise_source=noise_source)
    regen_seconds = time.perf_counter() - regen_started

    decode_seconds = {}
    started = time.perf_counter()
    pixels_base = er.decode_latents(pipeline, base.latents)
    decode_seconds["original"] = time.perf_counter() - started
    started = time.perf_counter()
    pixels_regen = er.decode_latents(pipeline, regenerated.latents)
    decode_seconds["full_regeneration"] = time.perf_counter() - started

    records = []
    for target in targets:
        case_dir = Path(output_dir) / f"{group_id}_chunk{target}"
        case_dir.mkdir(parents=True, exist_ok=True)
        entry = base.checkpoint_entries[target]
        checkpoint = ec.load_edit_checkpoint(entry["path"], verify_sha256=entry.get("sha256"))
        checkpoint = ec.checkpoint_to_device(checkpoint, device)
        base_chunk = base.latents[:, target * block:(target + 1) * block].clone()

        # ---- B2 / B3 / control --------------------------------------------
        replay = er.replay_chunk(pipeline, checkpoint, base_prompt, device=device,
                                 noise="from-cache", noise_source=noise_source,
                                 crossattn="restore", restore_rng=True)
        edit = er.replay_chunk(pipeline, checkpoint, edit_prompt, device=device,
                               noise="from-cache", noise_source=noise_source,
                               crossattn="reset", restore_rng=True)
        control = er.replay_chunk(pipeline, checkpoint, edit_prompt, device=device,
                                  noise="from-cache", noise_source=noise_source,
                                  crossattn="restore", restore_rng=True)

        # ---- assemble (Gate C: exact outside preservation pre-decode) ------
        replay_video = er.assemble_latents(base.latents, replay.latents, target, block)
        edit_video = er.assemble_latents(base.latents, edit.latents, target, block)
        control_video = er.assemble_latents(base.latents, control.latents, target, block)

        latents = {"same_prompt_replay": replay_video, "local_edit": edit_video,
                   "crossattn_control": control_video}
        pixels = {"original": pixels_base, "full_regeneration": pixels_regen}
        for name, value in latents.items():
            started = time.perf_counter()
            pixels[name] = er.decode_latents(pipeline, value)
            decode_seconds[name] = time.perf_counter() - started

        reference = pixels["original"]
        span = er.chunk_frame_slice(target, block, reference.shape[1])
        base_chunk_pixels = reference[0, span["pixel_start"]:span["pixel_end"]]
        responsiveness = {}
        for name in ("local_edit", "same_prompt_replay", "crossattn_control", "full_regeneration"):
            variant_chunk = pixels[name][0, span["pixel_start"]:span["pixel_end"]]
            responsiveness[name] = em.evaluate_probe(group["probe"], base_chunk_pixels, variant_chunk)

        boundary_base = em.boundary_latent(base.latents, target, block)
        boundary_edit = em.boundary_latent(edit_video, target, block)
        boundary_base_pixels = em.boundary_pixels(reference, target, block)
        boundary_edit_pixels = em.boundary_pixels(pixels["local_edit"], target, block)
        boundary = {
            "latent": {"base": boundary_base, "edited": boundary_edit,
                       "delta": em.boundary_delta(boundary_edit, boundary_base)},
            "pixel": {"base": boundary_base_pixels, "edited": boundary_edit_pixels,
                      "delta": em.boundary_delta(boundary_edit_pixels, boundary_base_pixels)},
        }
        if dino is not None:
            boundary["dino"] = _dino_boundary(dino, reference, pixels["local_edit"], span)

        outside_checked = er.outside_exact(edit_video, base.latents, target, block)
        outside_after_vae = 0.0
        for index in range(num_chunks):
            if index == target:
                continue
            piece = er.chunk_frame_slice(index, block, reference.shape[1])
            outside_after_vae = max(outside_after_vae, float(
                (reference[:, piece["pixel_start"]:piece["pixel_end"]]
                 - pixels["local_edit"][:, piece["pixel_start"]:piece["pixel_end"]]).abs().max().item()))

        partial_generation_seconds = edit.timings["restore_seconds"] + edit.timings["denoise_seconds"]
        cache_load_seconds = _timed_load(entry)
        full_end_to_end = regen_seconds + decode_seconds["full_regeneration"]
        partial_end_to_end = (partial_generation_seconds + cache_load_seconds
                              + decode_seconds["local_edit"])
        cost = {
            "full_generation_seconds": regen_seconds,
            "base_generation_seconds_including_cache_write": base_seconds,
            "partial_restore_seconds": edit.timings["restore_seconds"],
            "partial_denoise_seconds": edit.timings["denoise_seconds"],
            "partial_generation_seconds": partial_generation_seconds,
            "cache_load_seconds_cold": cache_load_seconds,
            "decode_seconds": decode_seconds["local_edit"],
            "partial_end_to_end_seconds": partial_end_to_end,
            "full_end_to_end_seconds": full_end_to_end,
            "full_generated_chunks": num_chunks,
            "partial_regenerated_chunks": 1,
            "peak_vram_bytes": max(edit.peak_vram_bytes, regenerated.peak_vram_bytes),
            "cache_bytes_per_chunk": checkpoint.cache_bytes(),
            "cache_disk_bytes": entry.get("bytes"),
        }
        cost["R_time_generation"] = partial_generation_seconds / regen_seconds
        cost["R_time_end_to_end"] = partial_end_to_end / full_end_to_end

        for name, tensor in pixels.items():
            emedia.write_video(case_dir / f"{name}.mp4", emedia.frames_to_uint8(tensor[0]), fps)
        emedia.write_comparison(case_dir / "comparison.mp4", pixels, fps,
                                labels={"original": "B0 original",
                                        "same_prompt_replay": "B2 replay P0",
                                        "local_edit": "B3 local edit P1",
                                        "crossattn_control": "control P1 + cached text K/V",
                                        "full_regeneration": "B1 full regen P1"})
        emedia.write_chunk_strip(case_dir / "edited_chunk_strip.mp4", pixels,
                                 span["pixel_start"], span["pixel_end"], fps)

        record = {
            "sample_id": f"{group_id}_chunk{target}",
            "prompt_id": group["prompt_id"],
            "base_prompt": base_prompt,
            "edit_prompt": edit_prompt,
            "prompt_hash_old": ec.sha256_text(base_prompt),
            "prompt_hash_new": ec.sha256_text(edit_prompt),
            "target_chunk": target,
            "num_chunks": num_chunks,
            "seed": seed,
            "noise_sha256": tensor_digest(noise),
            "noise_source": noise_source,
            "chunk_latent_sha256": {
                "original": tensor_digest(base_chunk),
                "same_prompt_replay": tensor_digest(replay.latents),
                "local_edit": tensor_digest(edit.latents),
                "crossattn_control": tensor_digest(control.latents),
                "full_regeneration": tensor_digest(
                    regenerated.latents[:, target * block:(target + 1) * block])},
            "replay_fidelity": em.latent_stats(base_chunk, replay.latents),
            "control_fidelity_to_replay": em.latent_stats(replay.latents, control.latents),
            "edit_vs_replay": em.latent_stats(replay.latents, edit.latents),
            "rng": _rng_check(checkpoint, edit.recorded_noise),
            "responsiveness": responsiveness,
            "preservation": {"outside_exact": True, "chunks_checked": outside_checked,
                             "outside_max_abs_after_vae": outside_after_vae,
                             "note": "Gate C is asserted on pre-decode latents; the VAE is causal, "
                                     "so any post-decode leak is reported here separately."},
            "boundary": boundary,
            "cache": {key: value for key, value in entry.items() if key != "checkpoint"},
            "cost": cost,
            "videos": {name: str(case_dir / f"{name}.mp4") for name in pixels} | {
                "comparison": str(case_dir / "comparison.mp4"),
                "edited_chunk_strip": str(case_dir / "edited_chunk_strip.mp4")},
        }
        ex.write_json(case_dir / "metrics.json", record)
        records.append(record)
        print(f"        chunk={target} S_proxy(local)={responsiveness['local_edit']['S_proxy']:+.6f} "
              f"replay_mse={record['replay_fidelity']['mse']:.3e} "
              f"R_time={cost['R_time_generation']:.3f}", flush=True)
    return records


def evaluate_gates(records, config) -> dict:
    thresholds = config["gates"]
    minimum_s = float(thresholds["edit"].get("min_s_proxy", 0.0))
    minimum_ratio = float(thresholds["edit"].get("min_full_regeneration_ratio", 0.0))
    edit_cases, preservation_cases, cost_cases = [], [], []
    for record in records:
        local = record["responsiveness"]["local_edit"]["S_proxy"]
        replay = record["responsiveness"]["same_prompt_replay"]["S_proxy"]
        control = record["responsiveness"]["crossattn_control"]["S_proxy"]
        full = record["responsiveness"]["full_regeneration"]["S_proxy"]
        directional = bool(local > minimum_s and local > replay and local > control)
        ratio = (local / full) if full > 0 else None
        edit_cases.append({"sample_id": record["sample_id"], "prompt_id": record["prompt_id"],
                           "target_chunk": record["target_chunk"],
                           "s_proxy_local_edit": local, "s_proxy_same_prompt": replay,
                           "s_proxy_control": control, "s_proxy_full_regeneration": full,
                           "ratio_to_full_regeneration": ratio, "passed": directional,
                           "strong": bool(directional and ratio is not None and ratio >= minimum_ratio)})
        preservation_cases.append({"sample_id": record["sample_id"],
                                   "outside_exact": record["preservation"]["outside_exact"],
                                   "chunks_checked": record["preservation"]["chunks_checked"]})
        cost = record["cost"]
        enough_chunks = record["num_chunks"] >= int(thresholds["cost"].get("min_chunks", 5))
        cost_cases.append({"sample_id": record["sample_id"], "num_chunks": record["num_chunks"],
                           "R_time_generation": cost["R_time_generation"],
                           "R_time_end_to_end": cost["R_time_end_to_end"],
                           "partial_generation_seconds": cost["partial_generation_seconds"],
                           "full_generation_seconds": cost["full_generation_seconds"],
                           "counted": enough_chunks,
                           "passed": (not enough_chunks) or
                                     cost["R_time_generation"] < float(thresholds["cost"]["max_time_ratio"])})
    successful = [case for case in edit_cases if case["passed"]]
    strong = [case for case in edit_cases if case["strong"]]
    counted = [case for case in cost_cases if case["counted"]]
    return {
        "gate_b_editability": {"passed": bool(successful), "successful_cases": len(successful),
                               "strong_cases": len(strong),
                               "min_full_regeneration_ratio": minimum_ratio,
                               "cases": edit_cases},
        "gate_c_preservation": {
            "passed": all(case["outside_exact"] for case in preservation_cases),
            "outside_exact_fraction": (sum(case["outside_exact"] for case in preservation_cases)
                                       / len(preservation_cases)) if preservation_cases else 0.0,
            "cases": preservation_cases},
        "gate_d_cost": {"passed": all(case["passed"] for case in counted) if counted else True,
                        "cases": cost_cases,
                        "max_time_ratio": float(thresholds["cost"]["max_time_ratio"]),
                        "counted_cases": len(counted)},
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/edit_ready_mvp.yaml"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/edit_ready_mvp/local_edit_smoke.json")
    parser.add_argument("--video-root", type=Path,
                        default=ROOT / "validation/edit_ready_mvp/videos_smoke")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cases", type=int)
    parser.add_argument("--prompt-ids", nargs="*")
    parser.add_argument("--targets", type=int, nargs="*")
    parser.add_argument("--seed-stride", type=int, default=0,
                        help="Offset added to this config's seed for a second-seed replication")
    parser.add_argument("--dino", action="store_true", help="Add frozen DINOv2 distances")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    arguments = parser.parse_args()
    if not arguments.reviewed:
        parser.error("GPU editing is a reviewed experiment; pass --reviewed after Gate A is approved")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = ex.read_config(arguments.config)
    device = ex.device_for(arguments.gpu)
    cache_dir = arguments.cache_dir or (ROOT / config["cache"]["root"] / "edit")
    cache_dir.mkdir(parents=True, exist_ok=True)
    arguments.video_root.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    pipeline = load_pipeline(config, device)
    identity = ex.run_provenance(config, arguments.config)
    dino = None
    if arguments.dino:
        dino = em.DinoFeatureDistance(ROOT / config["model"]["dino"], device=device)

    cases = ex.expand_cases(config, cases=arguments.cases, targets=arguments.targets,
                            prompt_ids=arguments.prompt_ids,
                            seed_stride=arguments.seed_stride)
    # Shard whole prompts, not (prompt, target) pairs: two shards must never
    # regenerate the same base video or write the same cache file.
    groups = ex.shard(ex.group_cases(cases), arguments.shard, arguments.shards)
    if not groups:
        raise SystemExit("No cases selected")

    records = []
    for group in groups:
        print(f"[edit] {group['prompt_id']} targets={group['targets']} seed={group['seed']}", flush=True)
        records.extend(run_group(pipeline, config, group, device, arguments.video_root,
                                 cache_dir, identity, dino=dino))

    gates = evaluate_gates(records, config)
    payload = {
        "experiment": "B_local_edit",
        "provenance": ex.run_provenance(config, arguments.config,
                                        extra={"shard": arguments.shard, "shards": arguments.shards,
                                               "video_root": str(arguments.video_root)}),
        "config": config,
        "gates": gates,
        "cases": records,
        "aggregate": {
            "cases": len(records),
            "replay_fidelity_mse_mean": _mean(r["replay_fidelity"]["mse"] for r in records),
            "s_proxy_local_edit_mean": _mean(r["responsiveness"]["local_edit"]["S_proxy"] for r in records),
            "s_proxy_same_prompt_mean": _mean(r["responsiveness"]["same_prompt_replay"]["S_proxy"] for r in records),
            "s_proxy_control_mean": _mean(r["responsiveness"]["crossattn_control"]["S_proxy"] for r in records),
            "s_proxy_full_regeneration_mean": _mean(r["responsiveness"]["full_regeneration"]["S_proxy"] for r in records),
            "left_boundary_delta_mean": _mean(r["boundary"]["latent"]["delta"]["delta_left_mse"] for r in records),
            "right_boundary_delta_mean": _mean(r["boundary"]["latent"]["delta"]["delta_right_mse"] for r in records),
            "R_time_generation_mean": _mean(r["cost"]["R_time_generation"] for r in records),
            "R_time_end_to_end_mean": _mean(r["cost"]["R_time_end_to_end"] for r in records),
            "cache_bytes_per_chunk": records[0]["cost"]["cache_bytes_per_chunk"] if records else None,
        },
    }
    ex.write_json(arguments.output, payload)
    print(json.dumps(payload["aggregate"], indent=2))
    print(json.dumps({key: value["passed"] for key, value in gates.items()}, indent=2))


if __name__ == "__main__":
    main()
