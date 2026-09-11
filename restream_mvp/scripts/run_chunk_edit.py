"""Experiment B -- prompt-changed local replay (Gates B/C/D) plus controls.

Per prompt pair the base video and the full-regeneration reference are computed
once; then for every target chunk ``k`` in ``edit.targets`` (default 0, 1, 4):

* **replay**              - ``S_{k-1}`` + ``P0``           (bit-exact reopen)
* **text_rebind**         - ``S_{k-1}`` + ``P1`` with the text K/V rebound (the MVP)
* **history_recache**     - unchanged ``P0`` latents re-cached under ``P1``, then ``P1``
* **reverse**             - ``P1`` history (full-regen checkpoint) + ``P0`` text
* **crossattn_control**   - ``P1`` text but the cached ``P0`` text K/V kept (no-op control)
* **full_regeneration**   - ``P1`` trajectory (upper reference)

Chunk 0 has no visual history, so ``text_rebind`` there must equal the
full-regeneration chunk 0 bit-for-bit: that is the calibration point of the
editability curve ``R_k = S_text_rebind(k) / S_full(k)``.

Timing uses one timer around the real user path (disk -> CPU -> GPU -> restore ->
encode -> generate -> assemble -> decode); nothing is stitched together from
separate measurements.

Gate B's automatic aggregate only covers ``evidence: directional`` prompts; the
``evidence: qualitative`` prompts (smile / zoom / rain) only report a
proxy-consistent change count and stay for human review.
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
from restream.history_gate import set_history_gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = ("replay", "text_rebind", "history_recache", "reverse",
            "crossattn_control", "full_regeneration")


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def _cuda_state(device):
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_rng_state(torch.device(device).index or 0).clone()


def _rng_check(checkpoint, recorded) -> dict:
    stored = checkpoint.denoise_noise
    if not stored or not recorded:
        return {"comparable": False}
    maximum = max(float((left.float().cpu() - right.float().cpu()).abs().max().item())
                  for left, right in zip(stored, recorded))
    return {"comparable": True, "steps": len(stored), "max_abs": maximum, "exact": maximum == 0.0}


def _load_entry(entry, device, expected_model_hash, expected_config_hash):
    checkpoint = ec.load_edit_checkpoint(entry["path"], verify_sha256=entry.get("sha256"))
    checkpoint = ec.checkpoint_to_device(checkpoint, device)
    ec.verify_checkpoint_identity(checkpoint, model_hash=expected_model_hash,
                                  config_digest=expected_config_hash)
    return checkpoint


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


def run_edit_user_path(pipeline, entry, target, config, device, base, edit_prompt,
                       expected_model_hash, expected_config_hash, drop_cache=True) -> dict:
    """One timer around the path an interactive user would actually take."""
    block = int(pipeline.num_frame_per_block)
    noise_source = config["generation"].get("noise_source", "global")
    started = time.perf_counter()

    if drop_cache:
        ex.drop_page_cache(entry["path"])
    disk_started = time.perf_counter()
    checkpoint = ec.load_edit_checkpoint(entry["path"], verify_sha256=entry.get("sha256"))
    disk_seconds = time.perf_counter() - disk_started

    transfer_started = time.perf_counter()
    checkpoint = ec.checkpoint_to_device(checkpoint, device)
    transfer_seconds = time.perf_counter() - transfer_started

    restore_started = time.perf_counter()
    ec.verify_checkpoint_identity(checkpoint, model_hash=expected_model_hash,
                                  config_digest=expected_config_hash)
    ec.restore_checkpoint(pipeline, checkpoint, device=device, reset_crossattn=True)
    restore_seconds = time.perf_counter() - restore_started

    result = er.replay_chunk(pipeline, checkpoint, edit_prompt, device=device,
                             noise="from-cache", noise_source=noise_source,
                             crossattn="reset", restore_rng=True, prepare_state=False,
                             expected_model_hash=expected_model_hash,
                             expected_config_hash=expected_config_hash)
    assembled = er.assemble_latents(base.latents, result.latents, target, block)
    pixels = er.decode_latents(pipeline, assembled)
    total_seconds = time.perf_counter() - started
    return {
        "latents": result.latents,
        "recorded_noise": result.recorded_noise,
        "pixels": pixels,
        "assembled": assembled,
        "timings": {
            "disk_load_seconds": disk_seconds,
            "host_to_device_seconds": transfer_seconds,
            "restore_seconds": restore_seconds,
            "prepare_seconds": result.timings.get("prepare_seconds", 0.0),
            "denoise_seconds": result.timings.get("denoise_seconds", 0.0),
            "bytes_loaded": entry.get("bytes"),
            "page_cache_dropped": bool(drop_cache and hasattr(os, "posix_fadvise")),
        },
        "end_to_end_seconds": total_seconds,
        "peak_vram_bytes": result.peak_vram_bytes,
    }


def run_group(pipeline, config, group, device, output_dir, cache_dir, identity, dino=None, history_gate_value=1.0) -> list:
    set_history_gate(pipeline, history_gate_value)
    block = int(pipeline.num_frame_per_block)
    seed = int(group["seed"])
    base_prompt, edit_prompt = group["base_prompt"], group["edit_prompt"]
    evidence = group.get("evidence", "directional")
    targets = [int(value) for value in group["targets"]]
    noise = ex.sample_noise(config, device, seed)
    num_chunks = noise.shape[1] // block
    for target in targets:
        if target >= num_chunks:
            raise ValueError(f"Target chunk {target} outside {num_chunks} chunks")
    noise_source = config["generation"].get("noise_source", "global")
    fps = ex.generation_geometry(config)["fps"]
    group_id = f"edit_{group['prompt_id']}_seed{seed}"
    regen_cache = Path(cache_dir).parent / f"{Path(cache_dir).name}_regen"

    # Every full generation re-seeds and re-draws its own initial noise.  Without
    # this the *second* generation would inherit the RNG stream left behind by the
    # first, so the full-regeneration reference would use different per-step
    # re-noising noise and the base/reference comparison would be noise-confounded
    # (this was caught by the chunk-0 calibration control).
    noise = ex.sample_noise(config, device, seed)

    # ---- B0: original generation, capturing the edit cache -----------------
    noise_base = ex.sample_noise(config, device, seed)
    rng_at_base_start = _cuda_state(device)
    base_started = time.perf_counter()
    base = er.stream_generate(
        pipeline, noise_base, base_prompt, sample_id=group_id,
        seed=seed, model_hash=identity["model_checkpoint_sha256"],
        model_record=identity["model_identity"], config_digest=identity["config_hash"],
        cache_dir=cache_dir, save_chunks=targets, noise_source=noise_source)
    base_seconds = time.perf_counter() - base_started

    # ---- B1: full regeneration from scratch with the new prompt -----------
    # Checkpoints are kept so the reverse control (P1 history + P0 text) can be
    # run from the *same* P1 trajectory.
    # The timed full-regeneration baseline writes no cache: a user who only wants
    # a fresh video does not pay for per-chunk checkpoints, and including that
    # cost would flatter the partial-edit ratio.
    noise_regen = ex.sample_noise(config, device, seed)
    rng_at_regen_start = _cuda_state(device)
    regen_started = time.perf_counter()
    regenerated = er.stream_generate(
        pipeline, noise_regen, edit_prompt, sample_id=f"{group_id}_fullregen", seed=seed,
        model_hash=identity["model_checkpoint_sha256"], model_record=identity["model_identity"],
        config_digest=identity["config_hash"], cache_dir=None, save_chunks=[],
        noise_source=noise_source)
    regen_seconds = time.perf_counter() - regen_started
    rng_stream_match = None
    noise_match = bool(torch.equal(noise_base, noise_regen))
    if rng_at_base_start is not None and rng_at_regen_start is not None:
        rng_stream_match = bool(torch.equal(rng_at_base_start, rng_at_regen_start))

    # Second regeneration whose checkpoints feed the reverse control (P1 history +
    # P0 text).  Not part of the reported cost.
    reverse_source = er.stream_generate(
        pipeline, ex.sample_noise(config, device, seed), edit_prompt,
        sample_id=f"{group_id}_fullregen_ckpt", seed=seed,
        model_hash=identity["model_checkpoint_sha256"], model_record=identity["model_identity"],
        config_digest=identity["config_hash"], cache_dir=regen_cache, save_chunks=targets,
        noise_source=noise_source)
    if not torch.equal(reverse_source.latents, regenerated.latents):
        raise RuntimeError("Checkpointed regeneration diverged from the timed baseline")

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
        regen_entry = reverse_source.checkpoint_entries[target]
        base_chunk = base.latents[:, target * block:(target + 1) * block].clone()
        regen_chunk = regenerated.latents[:, target * block:(target + 1) * block].clone()
        reference = pixels_base
        span = er.chunk_frame_slice(target, block, reference.shape[1])
        base_chunk_pixels = reference[0, span["pixel_start"]:span["pixel_end"]]
        regen_chunk_pixels = pixels_regen[0, span["pixel_start"]:span["pixel_end"]]

        # ---- replay: P0 history + P0 text ---------------------------------
        checkpoint = _load_entry(entry, device, identity["model_checkpoint_sha256"],
                                 identity["config_hash"])
        replay = er.replay_chunk(pipeline, checkpoint, base_prompt, device=device,
                                 noise="from-cache", noise_source=noise_source,
                                 crossattn="restore", restore_rng=True,
                                 expected_model_hash=identity["model_checkpoint_sha256"],
                                 expected_config_hash=identity["config_hash"])

        # ---- crossattn control: P0 history + P1 text + cached text K/V -----
        control = er.replay_chunk(pipeline, checkpoint, edit_prompt, device=device,
                                  noise="from-cache", noise_source=noise_source,
                                  crossattn="restore", restore_rng=True,
                                  expected_model_hash=identity["model_checkpoint_sha256"],
                                  expected_config_hash=identity["config_hash"])

        # ---- history recache: unchanged P0 latents, re-cached under P1 -----
        recache_started = time.perf_counter()
        er.recache_history(pipeline, base.latents, edit_prompt, target, device=device)
        recache_seconds = time.perf_counter() - recache_started
        cache_match = er.compare_caches(pipeline.kv_cache1, checkpoint.kv_cache)
        crossattn_match = er.compare_caches(pipeline.crossattn_cache, checkpoint.crossattn_cache,
                                            labels=("k", "v", "is_init"))
        recache = er.replay_chunk(pipeline, checkpoint, edit_prompt, device=device,
                                  noise="from-cache", noise_source=noise_source,
                                  crossattn="restore", restore_rng=True, prepare_state=False,
                                  expected_model_hash=identity["model_checkpoint_sha256"],
                                  expected_config_hash=identity["config_hash"])

        # ---- reverse: P1 history + P0 text (history-confound diagnostic) ---
        regen_checkpoint = _load_entry(regen_entry, device, identity["model_checkpoint_sha256"],
                                       identity["config_hash"])
        reverse = er.replay_chunk(pipeline, regen_checkpoint, base_prompt, device=device,
                                  noise="from-cache", noise_source=noise_source,
                                  crossattn="reset", restore_rng=True,
                                  expected_model_hash=identity["model_checkpoint_sha256"],
                                  expected_config_hash=identity["config_hash"])

        # ---- the timed user path (text rebind on the P0 history) -----------
        user_path = run_edit_user_path(pipeline, entry, target, config, device, base,
                                       edit_prompt, identity["model_checkpoint_sha256"],
                                       identity["config_hash"])

        # ---- assemble every variant ---------------------------------------
        assembled = {
            "replay": er.assemble_latents(base.latents, replay.latents, target, block),
            "text_rebind": user_path["assembled"],
            "history_recache": er.assemble_latents(base.latents, recache.latents, target, block),
            "reverse": er.assemble_latents(regenerated.latents, reverse.latents, target, block),
            "crossattn_control": er.assemble_latents(base.latents, control.latents, target, block),
            "full_regeneration": regenerated.latents,
        }
        pixels = {"original": pixels_base, "full_regeneration": pixels_regen,
                  "text_rebind": user_path["pixels"]}
        for name in ("replay", "history_recache", "reverse", "crossattn_control"):
            started = time.perf_counter()
            pixels[name] = er.decode_latents(pipeline, assembled[name])
            decode_seconds[name] = time.perf_counter() - started

        responsiveness = {}
        for name in ("replay", "text_rebind", "history_recache", "reverse",
                     "crossattn_control", "full_regeneration"):
            responsiveness[name] = em.evaluate_probe(
                group["probe"], base_chunk_pixels, pixels[name][0, span["pixel_start"]:span["pixel_end"]])

        # ---- R_k: how much of the full-regeneration response is recovered ---
        full_response = responsiveness["full_regeneration"]["S_proxy"]
        rebind_response = responsiveness["text_rebind"]["S_proxy"]
        ratio = (rebind_response / full_response) if full_response > 0 else None

        # ---- chunk-0 calibration: no history => must equal full regeneration
        chunk0_exact = None
        if target == 0:
            chunk0_exact = bool(torch.equal(user_path["latents"], regen_chunk))

        # ---- boundary continuity ------------------------------------------
        boundary_base = em.boundary_latent(base.latents, target, block)
        boundary_edit = em.boundary_latent(assembled["text_rebind"], target, block)
        boundary_base_pixels = em.boundary_pixels(reference, target, block)
        boundary_edit_pixels = em.boundary_pixels(pixels["text_rebind"], target, block)
        boundary = {
            "latent": {"base": boundary_base, "edited": boundary_edit,
                       "delta": em.boundary_delta(boundary_edit, boundary_base)},
            "pixel": {"base": boundary_base_pixels, "edited": boundary_edit_pixels,
                      "delta": em.boundary_delta(boundary_edit_pixels, boundary_base_pixels)},
        }
        if dino is not None:
            boundary["dino"] = _dino_boundary(dino, reference, pixels["text_rebind"], span)

        outside_checked = er.outside_exact(assembled["text_rebind"], base.latents, target, block)
        outside_after_vae = 0.0
        for index in range(num_chunks):
            if index == target:
                continue
            piece = er.chunk_frame_slice(index, block, reference.shape[1])
            outside_after_vae = max(outside_after_vae, float(
                (reference[:, piece["pixel_start"]:piece["pixel_end"]]
                 - pixels["text_rebind"][:, piece["pixel_start"]:piece["pixel_end"]]).abs().max().item()))

        timings = user_path["timings"]
        partial_generation_seconds = (timings["disk_load_seconds"] + timings["host_to_device_seconds"]
                                      + timings["restore_seconds"] + timings["prepare_seconds"]
                                      + timings["denoise_seconds"])
        partial_compute_seconds = (timings["restore_seconds"] + timings["prepare_seconds"]
                                   + timings["denoise_seconds"])
        full_end_to_end = regen_seconds + decode_seconds["full_regeneration"]
        cost = {
            "full_generation_seconds": regen_seconds,
            "base_generation_seconds_including_cache_write": base_seconds,
            "partial_end_to_end_seconds": user_path["end_to_end_seconds"],
            "partial_generation_seconds": partial_generation_seconds,
            "partial_compute_seconds": partial_compute_seconds,
            "disk_load_seconds": timings["disk_load_seconds"],
            "host_to_device_seconds": timings["host_to_device_seconds"],
            "restore_seconds": timings["restore_seconds"],
            "encode_seconds": timings["prepare_seconds"],
            "denoise_seconds": timings["denoise_seconds"],
            "decode_seconds": user_path["end_to_end_seconds"] - partial_generation_seconds,
            "page_cache_dropped": timings["page_cache_dropped"],
            "cache_bytes_per_chunk": checkpoint.cache_bytes(),
            "cache_disk_bytes": entry.get("bytes"),
            "recache_seconds": recache_seconds,
            "full_generated_chunks": num_chunks,
            "partial_regenerated_chunks": 1,
            "peak_vram_bytes": max(user_path["peak_vram_bytes"], regenerated.peak_vram_bytes),
        }
        cost["R_time_generation"] = partial_generation_seconds / regen_seconds
        cost["R_time_compute"] = partial_compute_seconds / regen_seconds
        cost["R_time_end_to_end"] = user_path["end_to_end_seconds"] / full_end_to_end

        videos = dict(pixels)
        for name, tensor in videos.items():
            emedia.write_video(case_dir / f"{name}.mp4", emedia.frames_to_uint8(tensor[0]), fps)
        emedia.write_comparison(case_dir / "comparison.mp4", videos, fps,
                                labels={"original": "B0 original P0",
                                        "replay": "replay P0 (exact)",
                                        "text_rebind": "TEXT REBIND P1 (MVP)",
                                        "history_recache": "history recache P1",
                                        "reverse": "reverse P1 hist + P0 text",
                                        "crossattn_control": "control P1 + cached text KV",
                                        "full_regeneration": "FULL REGEN P1"})
        emedia.write_chunk_strip(case_dir / "edited_chunk_strip.mp4", videos,
                                 span["pixel_start"], span["pixel_end"], fps)

        record = {
            "sample_id": f"{group_id}_chunk{target}",
            "prompt_id": group["prompt_id"],
            "evidence": evidence,
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
                "replay": tensor_digest(replay.latents),
                "text_rebind": tensor_digest(user_path["latents"]),
                "history_recache": tensor_digest(recache.latents),
                "reverse": tensor_digest(reverse.latents),
                "crossattn_control": tensor_digest(control.latents),
                "full_regeneration": tensor_digest(regen_chunk)},
            "replay_fidelity": em.latent_stats(base_chunk, replay.latents),
            "control_fidelity_to_replay": em.latent_stats(replay.latents, control.latents),
            "recache_fidelity_to_replay": em.latent_stats(replay.latents, recache.latents),
            "text_rebind_vs_replay": em.latent_stats(replay.latents, user_path["latents"]),
            "reverse_vs_full_regeneration": em.latent_stats(regen_chunk, reverse.latents),
            "recache_cache_matches_checkpoint": cache_match,
            "recache_crossattn_matches_checkpoint": crossattn_match,
            "recache_note": "recache rebuilds the cache from unchanged latents under the new "
                            "prompt; the self-attention K/V are prompt-independent, so this "
                            "should reproduce the checkpoint cache exactly.",
            "rng": _rng_check(checkpoint, user_path["recorded_noise"]),
            "responsiveness": responsiveness,
            "editability": {"S_text_rebind": rebind_response, "S_full_regeneration": full_response,
                            "R_k": ratio, "chunk0_matches_full_regeneration": chunk0_exact},
            "sanity": {
                "chunk0_text_rebind_equals_full_regeneration": chunk0_exact,
                "base_and_full_regen_share_rng_stream": rng_stream_match,
                "base_and_full_regen_share_initial_noise": noise_match,
                # At chunk 0 the checkpoint has no text binding yet, so keeping the
                # (empty) cross-attention cache is equivalent to resetting it; the
                # no-op control is only meaningful once a P0 binding exists.
                "control_equals_replay": bool(torch.equal(control.latents, replay.latents))
                if target > 0 else None,
                "control_equals_text_rebind": bool(
                    torch.equal(control.latents, user_path["latents"]))
                if target == 0 else None,
                "recache_equals_text_rebind": bool(
                    torch.equal(recache.latents, user_path["latents"])),
            },
            "preservation": {"outside_exact": True, "chunks_checked": outside_checked,
                             "outside_max_abs_after_vae": outside_after_vae,
                             "note": "Gate C is asserted on pre-decode latents; the VAE is causal, "
                                     "so any post-decode leak is reported here separately."},
            "boundary": boundary,
            "cache": {key: value for key, value in entry.items() if key != "checkpoint"},
            "cost": cost,
            "videos": {name: str(case_dir / f"{name}.mp4") for name in videos} | {
                "comparison": str(case_dir / "comparison.mp4"),
                "edited_chunk_strip": str(case_dir / "edited_chunk_strip.mp4")},
        }
        ex.write_json(case_dir / "metrics.json", record)
        records.append(record)
        print(f"        chunk={target} S(text_rebind)={rebind_response:+.6f} "
              f"R_k={'n/a' if ratio is None else f'{ratio:.3f}'} "
              f"chunk0_exact={chunk0_exact} recache_eq_rebind="
              f"{record['sanity']['recache_equals_text_rebind']} "
              f"R_e2e={cost['R_time_end_to_end']:.3f}", flush=True)
    return records


def evaluate_gates(records, config) -> dict:
    thresholds = config["gates"]
    minimum_s = float(thresholds["edit"].get("min_s_proxy", 0.0))
    minimum_ratio = float(thresholds["edit"].get("min_full_regeneration_ratio", 0.0))
    directional = [record for record in records if record.get("evidence") == "directional"]
    # Chunk 0 has no visual history: it calibrates the rebinding implementation
    # and must not be pooled with the committed-history edit cases.
    semantic = [record for record in directional if record["target_chunk"] > 0]
    calibration = [record for record in directional if record["target_chunk"] == 0]
    qualitative = [record for record in records if record.get("evidence") != "directional"]

    def case_entry(record):
        local = record["responsiveness"]["text_rebind"]["S_proxy"]
        replay = record["responsiveness"]["replay"]["S_proxy"]
        control = record["responsiveness"]["crossattn_control"]["S_proxy"]
        full = record["responsiveness"]["full_regeneration"]["S_proxy"]
        # The cached-text-K/V control is only a no-op once the checkpoint holds a
        # P0 binding; at chunk 0 the cache is uninitialised, so the "control"
        # legitimately equals the edit and must not veto the case.
        control_informative = record["target_chunk"] > 0
        directional = bool(local > minimum_s and local > replay
                           and (local > control or not control_informative))
        ratio = (local / full) if full > 0 else None
        return {"sample_id": record["sample_id"], "prompt_id": record["prompt_id"],
                "target_chunk": record["target_chunk"],
                "control_informative": control_informative,
                "s_proxy_text_rebind": local, "s_proxy_replay": replay,
                "s_proxy_control": control, "s_proxy_full_regeneration": full,
                "R_k": ratio, "passed": directional,
                "strong": bool(directional and ratio is not None and ratio >= minimum_ratio)}

    edit_cases = [case_entry(record) for record in semantic]
    calibration_cases = [case_entry(record) for record in calibration]
    qualitative_cases = [case_entry(record) for record in qualitative]
    preservation_cases = [{"sample_id": record["sample_id"],
                           "outside_exact": record["preservation"]["outside_exact"],
                           "chunks_checked": record["preservation"]["chunks_checked"]}
                          for record in records]
    cost_cases = []
    for record in records:
        cost = record["cost"]
        enough = record["num_chunks"] >= int(thresholds["cost"].get("min_chunks", 5))
        cost_cases.append({"sample_id": record["sample_id"], "num_chunks": record["num_chunks"],
                           "R_time_generation": cost["R_time_generation"],
                           "R_time_compute": cost["R_time_compute"],
                           "R_time_end_to_end": cost["R_time_end_to_end"],
                           "partial_generation_seconds": cost["partial_generation_seconds"],
                           "full_generation_seconds": cost["full_generation_seconds"],
                           "counted": enough,
                           "passed": (not enough) or
                                     cost["R_time_end_to_end"] < float(thresholds["cost"]["max_time_ratio"])})
    successful = [case for case in edit_cases if case["passed"]]
    strong = [case for case in edit_cases if case["strong"]]
    counted = [case for case in cost_cases if case["counted"]]
    by_chunk = {}
    for case in edit_cases:
        by_chunk.setdefault(case["target_chunk"], []).append(case)
    return {
        "gate_b_editability": {
            "passed": bool(successful), "semantic_cases": len(edit_cases),
            "successful_cases": len(successful), "strong_cases": len(strong),
            "min_full_regeneration_ratio": minimum_ratio,
            "cases": edit_cases,
            "calibration_cases": len(calibration_cases),
            "calibration_case_entries": calibration_cases,
            "calibration_note": "Chunk 0 has no visual history; text rebind there equals the "
                                "full-regeneration chunk bit-for-bit and calibrates R_k.",
            "qualitative_cases": len(qualitative_cases),
            "qualitative_case_entries": qualitative_cases,
            "qualitative_consistent_cases": sum(case["s_proxy_text_rebind"] > case["s_proxy_replay"]
                                                for case in qualitative_cases),
            "R_k_by_chunk": {str(chunk): [case["R_k"] for case in cases]
                             for chunk, cases in sorted(by_chunk.items())},
            "note": "Only `evidence: directional` prompts enter this aggregate; qualitative "
                    "prompts only report a proxy-consistent change count.",
        },
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
    parser.add_argument("--history-gate", type=float, default=1.0,
                        help="History contribution interpolation in [0,1]; diagnostic only")
    parser.add_argument("--manifest", type=Path,
                        help="Sweep manifest; validates prompt IDs and selects its gates/targets")
    arguments = parser.parse_args()
    if not arguments.reviewed:
        parser.error("GPU editing is a reviewed experiment; pass --reviewed after Gate A is approved")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = ex.read_config(arguments.config)
    if arguments.manifest:
        manifest = json.loads(arguments.manifest.read_text())
        entries = manifest.get("cases", [])
        if not entries:
            parser.error("manifest has no cases")
        ids = sorted({e["edit"] for e in entries})
        arguments.prompt_ids = ids
        arguments.targets = sorted({int(e["target_chunk"]) for e in entries})
        gates = sorted({float(e["history_gate"]) for e in entries})
        if len(gates) != 1:
            parser.error("run one history_gate per invocation; shard the manifest by gate")
        arguments.history_gate = gates[0]
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
        print(f"[edit] {group['prompt_id']} ({group.get('evidence')}) "
              f"targets={group['targets']} seed={group['seed']}", flush=True)
        records.extend(run_group(pipeline, config, group, device, arguments.video_root,
                                 cache_dir, identity, dino=dino,
                                 history_gate_value=arguments.history_gate))

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
            "recache_fidelity_to_replay_mse_mean": _mean(
                r["recache_fidelity_to_replay"]["mse"] for r in records),
            "s_proxy_text_rebind_mean": _mean(
                r["responsiveness"]["text_rebind"]["S_proxy"] for r in records),
            "s_proxy_replay_mean": _mean(r["responsiveness"]["replay"]["S_proxy"] for r in records),
            "s_proxy_control_mean": _mean(
                r["responsiveness"]["crossattn_control"]["S_proxy"] for r in records),
            "s_proxy_full_regeneration_mean": _mean(
                r["responsiveness"]["full_regeneration"]["S_proxy"] for r in records),
            "R_k_by_chunk": gates["gate_b_editability"]["R_k_by_chunk"],
            "left_boundary_delta_mean": _mean(
                r["boundary"]["latent"]["delta"]["delta_left_mse"] for r in records),
            "right_boundary_delta_mean": _mean(
                r["boundary"]["latent"]["delta"]["delta_right_mse"] for r in records),
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
