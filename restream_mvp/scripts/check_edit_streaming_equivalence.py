"""Cheap GPU gate before any experiment: does the mirrored chunk loop match upstream?

Two checks, both on a deliberately tiny video (default 6 latent frames = 2 AR
chunks) so this costs seconds instead of minutes:

1. ``stream_generate`` vs ``CausalInferencePipeline.inference`` - bit-exact?
2. capture a checkpoint before chunk 1, reopen it twice with the same prompt and
   confirm the two replays are bit-identical and that the recorded RNG state
   reproduces the recorded in-chunk noise draws.

This is the Step 1-4 evidence required by the plan before gate A is attempted on
real cases.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402
from restream.runtime import load_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/edit_ready_mvp.yaml"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/edit_ready_mvp/streaming_equivalence.json")
    parser.add_argument("--latent-frames", type=int, default=6)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--reviewed", action="store_true")
    arguments = parser.parse_args()
    if not arguments.reviewed:
        parser.error("GPU checks require --reviewed")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = ex.read_config(arguments.config)
    config["generation"]["num_latent_frames"] = int(arguments.latent_frames)
    device = ex.device_for(arguments.gpu)
    torch.set_grad_enabled(False)
    pipeline = load_pipeline(config, device)
    identity = ex.run_provenance(config, arguments.config)
    prompt = config["edit"]["prompts"][0]["base"]
    seed = int(config["seed"])

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    equivalence = er.verify_upstream_equivalence(pipeline, ex.sample_noise(config, device, seed), prompt)

    noise = ex.sample_noise(config, device, seed)
    result = er.stream_generate(pipeline, noise, prompt, sample_id="equivalence",
                                seed=seed, model_hash=identity["model_checkpoint_sha256"],
                                model_record=identity["model_identity"],
                                config_digest=identity["config_hash"], cache_dir=None,
                                save_chunks=None, noise_source="global")
    checkpoint = result.checkpoint_entries[1]["checkpoint"]
    block = int(pipeline.num_frame_per_block)
    base_chunk = result.latents[:, block:2 * block].clone()
    first = er.replay_chunk(pipeline, checkpoint, prompt, device=device, noise="from-cache",
                            noise_source="global", crossattn="restore", restore_rng=True)
    second = er.replay_chunk(pipeline, checkpoint, prompt, device=device, noise="from-cache",
                             noise_source="global", crossattn="restore", restore_rng=True)
    rng_exact = None
    if checkpoint.denoise_noise and first.recorded_noise:
        rng_exact = all(torch.equal(left, right)
                        for left, right in zip(checkpoint.denoise_noise, first.recorded_noise))

    payload = {
        "provenance": ex.run_provenance(config, arguments.config,
                                        extra={"latent_frames": arguments.latent_frames}),
        "upstream_equivalence": equivalence,
        "checkpoint_bytes": checkpoint.cache_bytes(),
        "cache_provenance": checkpoint.provenance,
        "replay_original_vs_replay": em.latent_stats(base_chunk, first.latents),
        "replay_repeat_baseline": em.latent_stats(first.latents, second.latents),
        "rng_draws_exact": rng_exact,
        "rng_steps": len(checkpoint.denoise_noise or []),
    }
    payload["passed"] = bool(equivalence["exact"] and rng_exact
                             and payload["replay_original_vs_replay"]["exact"])
    ex.write_json(arguments.output, payload)
    print(f"upstream exact={equivalence['exact']} max_abs={equivalence['max_abs_diff']:.3e}")
    print(f"replay exact={payload['replay_original_vs_replay']['exact']} "
          f"rng_exact={rng_exact} cache_bytes={checkpoint.cache_bytes()}")
    print(f"passed={payload['passed']}")
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
