"""Experiment A -- same-prompt single-chunk replay exactness (Gate A).

For every prompt case:

1. generate the whole video once, capturing an ``EditCheckpoint`` before each
   *target* AR chunk (the Generation-Time Edit Cache);
2. for every target chunk, reopen it from ``S_{k-1}`` with the original prompt and
   the original chunk noise, twice;
3. compare the reopened chunk with the original chunk, and the two replays with
   each other, to obtain the same-cache repeat-noise baseline;
4. verify that the recorded RNG state reproduces the exact in-chunk noise draws.

Gate A passes per case when the replay is bit-exact, or when its distance is
within ``relative_to_repeat`` x the repeat baseline and the cosine stays above
``cosine_floor``.  If Gate A fails, the plan says stop: do not run edit prompts.
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
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.runtime import load_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def noise_reproduction(checkpoint, recorded) -> dict:
    stored = checkpoint.denoise_noise
    if not stored or not recorded:
        return {"comparable": False, "reason": "no recorded in-chunk noise"}
    if len(stored) != len(recorded):
        return {"comparable": False, "reason": f"step count {len(stored)} vs {len(recorded)}"}
    values = [float((left.float().cpu() - right.float().cpu()).abs().max().item())
              for left, right in zip(stored, recorded)]
    return {"comparable": True, "steps": len(stored), "max_abs": max(values),
            "exact": all(value == 0.0 for value in values)}


def tensor_digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def replay_target(pipeline, prompt, target, base, device, noise_source, decode_pixels,
                  decoded_base=None, expected_model_hash=None, expected_config_hash=None) -> dict:
    block = int(pipeline.num_frame_per_block)
    entry = base.checkpoint_entries[target]
    checkpoint = entry.get("checkpoint") or ec.load_edit_checkpoint(
        entry["path"], verify_sha256=entry.get("sha256"))
    checkpoint = ec.checkpoint_to_device(checkpoint, device)
    base_chunk = base.latents[:, target * block:(target + 1) * block].clone()

    first = er.replay_chunk(pipeline, checkpoint, prompt, device=device, noise="from-cache",
                            noise_source=noise_source, crossattn="restore", restore_rng=True,
                            expected_model_hash=expected_model_hash,
                            expected_config_hash=expected_config_hash)
    second = er.replay_chunk(pipeline, checkpoint, prompt, device=device, noise="from-cache",
                             noise_source=noise_source, crossattn="restore", restore_rng=True,
                             expected_model_hash=expected_model_hash,
                             expected_config_hash=expected_config_hash)

    record = {
        "target_chunk": target,
        "chunk_latent_sha256": {"original": tensor_digest(base_chunk),
                                "replay_one": tensor_digest(first.latents),
                                "replay_two": tensor_digest(second.latents)},
        "latent": {"original_vs_replay": em.latent_stats(base_chunk, first.latents),
                   "repeat_baseline": em.latent_stats(first.latents, second.latents)},
        "rng": noise_reproduction(checkpoint, first.recorded_noise),
        "checkpoint": {key: value for key, value in entry.items() if key != "checkpoint"},
        "timings": {"replay_restore_seconds": first.timings.get("restore_seconds"),
                    "replay_denoise_seconds": first.timings.get("denoise_seconds")},
        "peak_vram_bytes": first.peak_vram_bytes,
    }
    if decode_pixels and decoded_base is not None:
        assembled = er.assemble_latents(base.latents, first.latents, target, block)
        decoded_replay = er.decode_latents(pipeline, assembled)
        span = er.chunk_frame_slice(target, block, decoded_base.shape[1])
        record["pixel"] = em.pixel_stats(decoded_base[:, span["pixel_start"]:span["pixel_end"]],
                                         decoded_replay[:, span["pixel_start"]:span["pixel_end"]])
        worst = 0.0
        for index in range(base.latents.shape[1] // block):
            if index == target:
                continue
            piece = er.chunk_frame_slice(index, block, decoded_base.shape[1])
            worst = max(worst, float((decoded_base[:, piece["pixel_start"]:piece["pixel_end"]]
                                      - decoded_replay[:, piece["pixel_start"]:piece["pixel_end"]]
                                      ).abs().max().item()))
        record["pixel"]["outside_max_abs_after_vae"] = worst
        record["pixel"]["outside_note"] = (
            "Outside preservation is asserted pre-decode (Gate C); this value is the "
            "causal VAE temporal-propagation leak after full-video decode.")
    return record


def gate_a(records, gate_config) -> dict:
    relative = float(gate_config.get("relative_to_repeat", 3.0))
    epsilon = float(gate_config.get("epsilon", 1e-12))
    cosine_floor = float(gate_config.get("cosine_floor", 0.999))
    cases = []
    for record in records:
        original = record["latent"]["original_vs_replay"]
        repeat = record["latent"]["repeat_baseline"]
        if original["exact"]:
            mode, passed = "exact", True
        else:
            within = original["mse"] <= relative * repeat["mse"] + epsilon
            passed = bool(within and original["cosine"] >= cosine_floor)
            mode = "within_repeat_noise" if passed else "drift"
        cases.append({"sample_id": record["sample_id"], "prompt_id": record["prompt_id"],
                      "target_chunk": record["target_chunk"], "passed": passed, "mode": mode,
                      "latent_mse": original["mse"], "repeat_mse": repeat["mse"],
                      "cosine": original["cosine"], "max_abs": original["max_abs"],
                      "rng_exact": record["rng"].get("exact")})
    passed = bool(cases) and all(case["passed"] for case in cases)
    return {"passed": passed, "relative_to_repeat": relative, "epsilon": epsilon,
            "cosine_floor": cosine_floor, "cases": cases,
            "exact_cases": sum(case["mode"] == "exact" for case in cases),
            "total_cases": len(cases),
            "latent_mse_mean": sum(case["latent_mse"] for case in cases) / len(cases) if cases else None,
            "repeat_mse_mean": (sum(case["repeat_mse"] for case in cases) / len(cases)) if cases else None,
            "rng_exact_cases": sum(bool(case["rng_exact"]) for case in cases)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/edit_ready_mvp.yaml"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/edit_ready_mvp/replay_smoke.json")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cases", type=int, help="Limit the number of prompt cases")
    parser.add_argument("--prompt-ids", nargs="*")
    parser.add_argument("--targets", type=int, nargs="*")
    parser.add_argument("--seed-stride", type=int, default=0,
                        help="Offset added to this config's seed for a second-seed replication")
    parser.add_argument("--all-chunks", action="store_true",
                        help="Persist every chunk boundary instead of only the targets")
    parser.add_argument("--noise-source", default=None, choices=["global", "generator"])
    parser.add_argument("--no-decode", action="store_true", help="Skip VAE pixel comparison")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    arguments = parser.parse_args()
    if not arguments.reviewed:
        parser.error("GPU replay is a reviewed experiment; pass --reviewed after the protocol is approved")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = ex.read_config(arguments.config)
    device = ex.device_for(arguments.gpu)
    cache_dir = arguments.cache_dir or (ROOT / config["cache"]["root"] / "replay")
    cache_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    pipeline = load_pipeline(config, device)
    identity = ex.run_provenance(config, arguments.config)
    noise_source = arguments.noise_source or config["generation"].get("noise_source", "global")
    block = int(pipeline.num_frame_per_block)

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
        seed = int(group["seed"])
        noise = ex.sample_noise(config, device, seed)
        num_chunks = noise.shape[1] // block
        for target in group["targets"]:
            if target >= num_chunks:
                raise ValueError(f"Target chunk {target} outside {num_chunks} chunks")
        sample_id = f"replay_{group['prompt_id']}_seed{seed}"
        started = time.perf_counter()
        base = er.stream_generate(
            pipeline, noise, group["base_prompt"], sample_id=sample_id, seed=seed,
            model_hash=identity["model_checkpoint_sha256"], model_record=identity["model_identity"],
            config_digest=identity["config_hash"], cache_dir=cache_dir,
            save_chunks=None if arguments.all_chunks else group["targets"],
            noise_source=noise_source)
        generate_seconds = time.perf_counter() - started
        decoded_base = None if arguments.no_decode else er.decode_latents(pipeline, base.latents)
        for target in group["targets"]:
            print(f"[replay] {group['prompt_id']} chunk={target} seed={seed}", flush=True)
            record = replay_target(pipeline, group["base_prompt"], target, base, device,
                                   noise_source, not arguments.no_decode, decoded_base,
                                   identity["model_checkpoint_sha256"], identity["config_hash"])
            record.update({
                "sample_id": f"{sample_id}_chunk{target}",
                "prompt_id": group["prompt_id"],
                "prompt": group["base_prompt"],
                "prompt_hash": ec.sha256_text(group["base_prompt"]),
                "seed": seed,
                "num_chunks": num_chunks,
                "noise_sha256": tensor_digest(noise),
                "generate_seconds_including_cache_write": generate_seconds,
                "peak_vram_bytes": max(base.peak_vram_bytes, record["peak_vram_bytes"]),
            })
            records.append(record)
            original = record["latent"]["original_vs_replay"]
            print(f"        exact={original['exact']} mse={original['mse']:.3e} "
                  f"repeat_mse={record['latent']['repeat_baseline']['mse']:.3e} "
                  f"rng_exact={record['rng'].get('exact')}", flush=True)

    gate = gate_a(records, config["gates"]["replay"])
    payload = {
        "experiment": "A_same_prompt_replay",
        "provenance": ex.run_provenance(config, arguments.config,
                                        extra={"shard": arguments.shard, "shards": arguments.shards,
                                               "all_chunks": arguments.all_chunks,
                                               "noise_source": noise_source}),
        "config": config,
        "gate": gate,
        "cases": records,
    }
    ex.write_json(arguments.output, payload)
    print(json.dumps({key: value for key, value in gate.items() if key != "cases"}, indent=2))
    if not gate["passed"]:
        print("Gate A FAILED -- per the plan, stop before prompt-changed edits.", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
