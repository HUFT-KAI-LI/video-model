"""Fixed-history mechanism probe: generate B0/FullReg at g=1 once per prompt/seed.

Replay paired P0/P1 chunks from the same checkpoint at each requested gate.
Report E=S(P1,g)-S(P0,g), R=E/S_full and P0 drift; timing is invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_media as emedia  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.runtime import load_pipeline  # noqa: E402
from restream.history_release import manifest_groups, validate_smoke, load_sealed
from restream.history_gate import history_gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def _rng_check(checkpoint, recorded) -> dict:
    stored = checkpoint.denoise_noise
    if not stored or not recorded:
        return {"comparable": False}
    if len(stored) != len(recorded):
        return {"comparable": True, "exact": False, "reason": "noise step count mismatch"}
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


def run_group(pipeline, config, group, device, output_dir, cache_dir, identity,
              dino=None, history_gate_value=1.0, sealed=None) -> list:
    """Generate fixed g=1 references once, then intervene only during replay."""
    with history_gate(pipeline, 1.0):
        return _run_fixed_group(pipeline, config, group, device, output_dir, cache_dir,
                                identity, dino, history_gate_value, sealed)


def _run_fixed_group(pipeline, config, group, device, output_dir, cache_dir,
                     identity, dino, history_gate_value, sealed):
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
    regenerated = er.stream_generate(pipeline, ex.sample_noise(config, device, seed), edit_prompt,
                                     sample_id=f"{group_id}_fullregen", cache_dir=None,
                                     save_chunks=[], **generation_args)
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
        gates = group.get("gates_by_target", {}).get(target, [history_gate_value])
        for gate in sorted(gates, reverse=True):
            case_id = f"{group_id}_chunk{target}_g{gate:g}"
            case_dir = Path(output_dir) / case_id
            case_dir.mkdir(parents=True, exist_ok=False)
            results, assembled, pixels, rng = {}, {}, {}, {}
            for name, prompt, binding in (("replay", base_prompt, "restore"),
                                           ("text_rebind", edit_prompt, "reset")):
                # Reload for each member of the pair: no mutated state is shared.
                checkpoint = _load_entry(entry, device, identity["model_checkpoint_sha256"],
                                         identity["config_hash"])
                with history_gate(pipeline, gate):
                    result = er.replay_chunk(
                        pipeline, checkpoint, prompt, device=device, noise="from-cache",
                        noise_source=noise_source, crossattn=binding, restore_rng=True,
                        expected_model_hash=identity["model_checkpoint_sha256"],
                        expected_config_hash=identity["config_hash"])
                results[name] = result.latents
                rng[name] = _rng_check(checkpoint, result.recorded_noise)
                if rng[name].get("comparable") and not rng[name]["exact"]:
                    raise AssertionError(f"{case_id}: {name} replay noise differs from checkpoint")
                assembled[name] = er.assemble_latents(base.latents, result.latents, target, block)
                er.outside_exact(assembled[name], base.latents, target, block)
                pixels[name] = er.decode_latents(pipeline, assembled[name])
            drift = em.latent_stats(base_chunk, results["replay"])
            if gate == 1.0 or target == 0:
                if not drift["exact"]:
                    raise AssertionError(f"{case_id}: P0 must be exact base")
            if target == 0 and not torch.equal(results["text_rebind"], full_chunk):
                raise AssertionError(f"{case_id}: chunk 0 P1 must equal full regeneration")
            digests = {name: tensor_digest(value) for name, value in results.items()}
            digests.update(original=tensor_digest(base_chunk),
                           full_regeneration=tensor_digest(full_chunk))
            sealed_exact = None
            if gate == 1.0 and sealed is not None:
                expected = sealed[(group["prompt_id"], seed, target)]
                for name in ("original", "text_rebind", "full_regeneration"):
                    if digests[name] != expected["chunk_latent_sha256"][name]:
                        raise AssertionError(f"{case_id}: sealed {name} digest mismatch")
                sealed_exact = True
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
                    boundary[name]["dino"] = _dino_boundary(dino, pixels_base, pixels[name], span)
            outside_pixels = {}
            for name in results:
                outside_pixels[name] = max(
                    [float((pixels_base[:, :sl.start] - pixels[name][:, :sl.start]).abs().max())
                     if sl.start else 0.0,
                     float((pixels_base[:, sl.stop:] - pixels[name][:, sl.stop:]).abs().max())
                     if sl.stop < pixels_base.shape[1] else 0.0])
            record = {
                "protocol": "fixed_history_paired_v2", "sample_id": case_id,
                "prompt_id": group["prompt_id"], "seed": seed, "target_chunk": target,
                "history_gate": gate, "reference_history_gate": 1.0,
                "policies": {"replay": "P0", "text_rebind": "P1"},
                "evidence": group.get("evidence", "directional"), "num_chunks": num_chunks,
                "base_prompt": base_prompt, "edit_prompt": edit_prompt,
                "noise_sha256": tensor_digest(noise), "noise_source": noise_source,
                "chunk_latent_sha256": digests, "rng": rng,
                "responsiveness": responsiveness, "editability": scores,
                "D_drift": drift["mse"], "D_drift_metric": "target_chunk_latent_mse",
                "drift": {"latent": drift,
                          "pixel": em.pixel_stats(base_frames, pixels["replay"][0, sl])},
                "sanity": {"g1_P0_exact_base": drift["exact"] if gate == 1.0 else None,
                           "g1_P1_exact_sealed": sealed_exact,
                           "chunk0_gate_invariant": True if target == 0 else None},
                "preservation": {"outside_exact": True, "policies_checked": ["P0", "P1"],
                                 "chunks_checked": num_chunks - 1,
                                 "outside_max_abs_after_vae": outside_pixels,
                                 "identity": appearance},
                "boundary": boundary,
                "cache": {key: value for key, value in entry.items() if key != "checkpoint"},
                "cost": {"status": "invalid_diagnostic", "passed": None,
                         "reason": "Mechanism probe: intermediate gates call attention twice; "
                                   "method timing is disabled for every gate."},
            }
            videos = {"original": pixels_base, "full_regeneration": pixels_full, **pixels}
            for name, tensor in videos.items():
                emedia.write_video(case_dir / f"{name}.mp4", emedia.frames_to_uint8(tensor[0]), fps)
            emedia.write_comparison(case_dir / "comparison.mp4", videos, fps,
                                    labels={"original": "B0 g=1", "full_regeneration": "FullReg P1 g=1",
                                            "replay": f"P0 g={gate:g}", "text_rebind": f"P1 g={gate:g}"})
            record["videos"] = {name: str(case_dir / f"{name}.mp4") for name in videos}
            record["videos"]["comparison"] = str(case_dir / "comparison.mp4")
            ex.write_json(case_dir / "metrics.json", record)
            records.append(record)
            print(f"[pair] {case_id} E={scores['E']:+.6f} R={scores['R_k']} "
                  f"D_drift={drift['mse']:.6g}", flush=True)
    return records


def paired_editability(response):
    p0 = response["replay"]["S_proxy"]
    p1 = response["text_rebind"]["S_proxy"]
    full = response["full_regeneration"]["S_proxy"]
    effect = p1 - p0
    return {"S_P0": p0, "S_P1": p1, "S_full": full, "E": effect,
            "R_k": effect / full if full > 0 else None,
            "ratio_status": "valid" if full > 0 else "nonpositive_full_reference"}


def evaluate_paired_gates(records, config):
    threshold = config["gates"]["edit"]
    rows = []
    for record in records:
        scores = paired_editability(record["responsiveness"])
        rows.append({key: record[key] for key in
                     ("sample_id", "prompt_id", "seed", "target_chunk", "history_gate", "D_drift")}
                    | scores | {"boundary": record["boundary"],
                                "preservation": record["preservation"],
                                "evidence": record["evidence"]})
    semantic = [row for row in rows if row["target_chunk"] > 0 and row["evidence"] == "directional"]
    successful = [row for row in semantic if row["E"] > float(threshold.get("min_s_proxy", 0))]
    return {
        "gate_b_editability": {"passed": bool(successful), "semantic_cases": len(semantic),
            "successful_cases": len(successful),
            "strong_cases": sum(row["R_k"] is not None and row["R_k"] >=
                                float(threshold.get("min_full_regeneration_ratio", 0))
                                for row in successful),
            "frontier": rows,
            "note": "Paired E=S(P1,g)-S(P0,g); chunk 0 calibrates, qualitative cases need review."},
        "gate_c_preservation": {"passed": all(r["preservation"]["outside_exact"] for r in records)},
        "gate_d_cost": {"passed": None, "status": "invalid_diagnostic", "counted_cases": 0,
                        "reason": "Mechanism-only probe; intermediate gates invoke attention twice."}}


def evaluate_gates(records, config) -> dict:
    if any(r.get("protocol") == "fixed_history_paired_v2" for r in records):
        if not all(r.get("protocol") == "fixed_history_paired_v2" for r in records):
            raise ValueError("Cannot mix legacy and paired protocols")
        return evaluate_paired_gates(records, config)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/edit_ready_mvp.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cases", type=int)
    parser.add_argument("--prompt-ids", nargs="*")
    parser.add_argument("--targets", type=int, nargs="*")
    parser.add_argument("--seed-stride", type=int, default=0)
    parser.add_argument("--dino", action="store_true", help="Record local frozen DINOv2 appearance proxies")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--history-gate", type=float, default=None)
    parser.add_argument("--manifest", type=Path, help="Schema 2 paired intervention manifest")
    parser.add_argument("--sealed-reference", type=Path, nargs="+",
                        help="Previous sealed JSON runs; compare original/P1/full chunk digests at g=1")
    parser.add_argument("--invariant-smoke", action="store_true",
                        help="Require 9 pairs / 18 replays and a sealed reference; abort on any invariant failure")
    arguments = parser.parse_args()
    if not arguments.reviewed:
        parser.error("GPU editing is a reviewed experiment; pass --reviewed after Gate A is approved")
    config = ex.read_config(arguments.config)
    if arguments.manifest:
        if (arguments.cases is not None or arguments.prompt_ids is not None or
            arguments.targets is not None or arguments.seed_stride != 0 or arguments.history_gate is not None):
            parser.error("manifest is authoritative; do not combine with case/seed/target/gate overrides")
        manifest = json.loads(arguments.manifest.read_text())
    else:
        gate = 1.0 if arguments.history_gate is None else arguments.history_gate
        cases = ex.expand_cases(config, cases=arguments.cases, targets=arguments.targets,
                                prompt_ids=arguments.prompt_ids, seed_stride=arguments.seed_stride)
        manifest = {"schema": 2, "cases": [
            {"edit": c["prompt_id"], "seed": c["seed"], "target_chunk": c["target_chunk"],
             "history_gate": gate} for c in cases]}
    try:
        groups = manifest_groups(config, manifest)
        if arguments.invariant_smoke:
            validate_smoke(groups)
            if not arguments.sealed_reference or arguments.shards != 1:
                parser.error("invariant smoke requires --sealed-reference and a single shard")
        groups = ex.shard(groups, arguments.shard, arguments.shards)
    except ValueError as error:
        parser.error(str(error))
    if not groups:
        parser.error("No cases selected")
    # A unique directory also prevents repeated identical runs from overwriting evidence.
    import tempfile
    root = ROOT / "validation/history_release"
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"paired_shard{arguments.shard}_", dir=root))
    output = arguments.output or run_dir / "results.json"
    video_root = arguments.video_root or run_dir / "videos"
    cache_dir = arguments.cache_dir or run_dir / "cache"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    for path in (video_root, cache_dir):
        if path.exists() and any(path.iterdir()):
            parser.error(f"Output directory must be empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = ex.device_for(arguments.gpu)
    if device.type != "cuda":
        parser.error("GPU runner requires working CUDA; CPU unit tests are not GPU smoke evidence")
    torch.set_grad_enabled(False)
    identity = ex.run_provenance(config, arguments.config, extra={
        "protocol": "fixed_history_paired_v2", "reference_history_gate": 1.0,
        "history_gates": sorted({gate for g in groups for values in g["gates_by_target"].values()
                                 for gate in values}, reverse=True),
        "manifest": manifest, "shard": arguments.shard, "shards": arguments.shards,
        "video_root": str(video_root), "cache_dir": str(cache_dir),
        "timing_status": "invalid_diagnostic"})
    sealed, sources = None, []
    if arguments.sealed_reference:
        sealed, sources = load_sealed(arguments.sealed_reference, groups, identity)
    identity["sealed_references"] = sources
    pipeline = load_pipeline(config, device)
    dino = em.DinoFeatureDistance(ROOT / config["model"]["dino"], device=device) if arguments.dino else None
    records = []
    try:
        for group in groups:
            records.extend(run_group(pipeline, config, group, device, video_root,
                                     cache_dir, identity, dino=dino, sealed=sealed))
    except Exception as error:
        ex.write_json(output, {"experiment": "history_release_paired", "status": "failed",
                              "provenance": identity, "cases": records,
                              "error": f"{type(error).__name__}: {error}",
                              "invariant_smoke": {"passed": False}})
        raise
    gates = evaluate_gates(records, config)
    payload = {"experiment": "history_release_paired", "schema": 2, "status": "complete",
               "provenance": identity, "config": config, "gates": gates, "cases": records,
               "invariant_smoke": {"requested": arguments.invariant_smoke,
                                   "passed": True if arguments.invariant_smoke else None,
                                   "pairs": len(records), "chunk_replays": 2 * len(records)},
               "note": "Mechanism results only; full sweep still requires review of GPU smoke."}
    ex.write_json(output, payload)
    print(f"Results: {output}")
    print(json.dumps({key: value["passed"] for key, value in gates.items()}, indent=2))


if __name__ == "__main__":
    main()
