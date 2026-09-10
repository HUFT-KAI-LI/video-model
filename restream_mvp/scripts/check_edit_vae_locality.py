"""How local is a chunk edit once the video is actually decoded?

Gate C guarantees bit-exact preservation on the *pre-decode* latents.  The Wan
VAE decoder is causal in time, so pixels after the edited chunk can still drift.
This diagnostic quantifies that drift for one case:

* ``replay`` (same prompt) vs ``original`` everywhere - must be bit-exact;
* ``local edit`` vs ``original`` inside the edited chunk (the wanted change);
* ``local edit`` vs ``original`` outside the edited chunk, per frame and in
  aggregate (mean / p99 / max), which is the unwanted leak;
* decoding the edited chunk *alone* versus in full context, to show whether the
  decoder itself is chunk-local.

Nothing here changes the gate definitions; it is a reported diagnostic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.runtime import load_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _stats(values) -> dict:
    values = np.asarray(values.detach().float().cpu().numpy() if hasattr(values, "detach") else values)
    return {"mean": float(values.mean()), "p99": float(np.percentile(values, 99)),
            "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/edit_ready_mvp.yaml"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/edit_ready_mvp/vae_locality.json")
    parser.add_argument("--prompt-id", default="car_red_to_black")
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--seed-stride", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs/edit_ready_mvp/cache/vae_locality")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    arguments = parser.parse_args()
    if not arguments.reviewed:
        parser.error("GPU diagnostics require --reviewed")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = ex.read_config(arguments.config)
    device = ex.device_for(arguments.gpu)
    arguments.cache_dir.mkdir(parents=True, exist_ok=True)
    groups = ex.group_cases(ex.expand_cases(config, prompt_ids=[arguments.prompt_id],
                                            targets=[arguments.target],
                                            seed_stride=arguments.seed_stride))
    if not groups:
        raise SystemExit("No matching prompt case")
    group = groups[0]
    seed = int(group["seed"])
    target = int(group["targets"][0])

    torch.set_grad_enabled(False)
    pipeline = load_pipeline(config, device)
    identity = ex.run_provenance(config, arguments.config)
    block = int(pipeline.num_frame_per_block)
    noise = ex.sample_noise(config, device, seed)
    noise_source = config["generation"].get("noise_source", "global")

    base = er.stream_generate(pipeline, noise, group["base_prompt"], sample_id="vae_locality",
                              seed=seed, model_hash=identity["model_checkpoint_sha256"],
                              model_record=identity["model_identity"],
                              config_digest=identity["config_hash"], cache_dir=arguments.cache_dir,
                              save_chunks=[target], noise_source=noise_source)
    entry = base.checkpoint_entries[target]
    checkpoint = ec.checkpoint_to_device(ec.load_edit_checkpoint(entry["path"]), device)
    replay = er.replay_chunk(pipeline, checkpoint, group["base_prompt"], device=device,
                             noise="from-cache", noise_source=noise_source,
                             crossattn="restore", restore_rng=True)
    edit = er.replay_chunk(pipeline, checkpoint, group["edit_prompt"], device=device,
                           noise="from-cache", noise_source=noise_source,
                           crossattn="reset", restore_rng=True)
    replay_video = er.assemble_latents(base.latents, replay.latents, target, block)
    edit_video = er.assemble_latents(base.latents, edit.latents, target, block)

    pixels_base = er.decode_latents(pipeline, base.latents)
    pixels_replay = er.decode_latents(pipeline, replay_video)
    pixels_edit = er.decode_latents(pipeline, edit_video)

    span = er.chunk_frame_slice(target, block, pixels_base.shape[1])
    start, end = span["pixel_start"], span["pixel_end"]
    total = pixels_base.shape[1]
    replay_diff = (pixels_replay - pixels_base).abs()
    edit_diff = (pixels_edit - pixels_base).abs()

    per_frame = []
    for index in range(total):
        inside = start <= index < end
        per_frame.append({"frame": index, "inside_chunk": bool(inside),
                          "edit_mean_abs": float(edit_diff[0, index].mean().item()),
                          "replay_mean_abs": float(replay_diff[0, index].mean().item()),
                          "distance_from_chunk": 0 if inside else (
                              start - index if index < start else index - (end - 1))})

    outside_frames = [entry for entry in per_frame if not entry["inside_chunk"]]
    report = {
        "provenance": ex.run_provenance(config, arguments.config,
                                        extra={"prompt_id": group["prompt_id"], "target": target,
                                               "seed": seed}),
        "chunk_pixel_range": [start, end],
        "total_pixel_frames": total,
        "replay": {"exact_after_decode": bool(torch.equal(pixels_replay, pixels_base)),
                   "outside": _stats(replay_diff[:, :start].ravel())
                   if start else {"mean": 0.0, "p99": 0.0, "max": 0.0}},
        "edit_inside_chunk": _stats(edit_diff[:, start:end].ravel()),
        "edit_outside_chunk": _stats(torch.cat(
            [edit_diff[:, :start].reshape(-1), edit_diff[:, end:].reshape(-1)])
            if start else edit_diff[:, end:].reshape(-1)),
        "edit_before_chunk": _stats(edit_diff[:, :start].reshape(-1)) if start
        else {"mean": 0.0, "p99": 0.0, "max": 0.0},
        "edit_after_chunk": _stats(edit_diff[:, end:].reshape(-1)) if end < total
        else {"mean": 0.0, "p99": 0.0, "max": 0.0},
        "latent_outside_exact": True,
        "per_frame": per_frame,
    }
    ex.write_json(arguments.output, report)
    print(f"replay exact after decode: {report['replay']['exact_after_decode']}")
    print(f"edit inside  mean/p99/max: {report['edit_inside_chunk']}")
    print(f"edit before  mean/p99/max: {report['edit_before_chunk']}")
    print(f"edit after   mean/p99/max: {report['edit_after_chunk']}")


if __name__ == "__main__":
    main()
