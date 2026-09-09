"""Compare full-video, prefix-video, and single-frame Wan VAE latents."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream.dataset import VideoDataset
from restream.runtime import load_pipeline, read_config


def compare(reference, candidate):
    reference, candidate = reference.float(), candidate.float()
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise RuntimeError("Non-finite VAE latent")
    flat_ref, flat_candidate = reference.flatten(), candidate.flatten()
    return {
        "mse": float((reference - candidate).square().mean().cpu()),
        "cosine": float(torch.nn.functional.cosine_similarity(flat_ref[None], flat_candidate[None]).cpu()),
        "reference_mean": float(reference.mean().cpu()),
        "candidate_mean": float(candidate.mean().cpu()),
        "reference_std": float(reference.std().cpu()),
        "candidate_std": float(candidate.std().cpu()),
        "reference_channel_mean": reference.mean(dim=(0, 1, 3, 4)).cpu().tolist(),
        "candidate_channel_mean": candidate.mean(dim=(0, 1, 3, 4)).cpu().tolist(),
        "reference_channel_std": reference.std(dim=(0, 1, 3, 4)).cpu().tolist(),
        "candidate_channel_std": candidate.std(dim=(0, 1, 3, 4)).cpu().tolist(),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/restream_mvp.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/anchor_latent_alignment.json")
    parser.add_argument("--cases", type=int, default=8)
    args = parser.parse_args()
    if args.cases <= 0:
        parser.error("--cases must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the VAE alignment check")
    config = read_config(args.config)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda", torch.cuda.current_device())
    pipeline = load_pipeline(config, device)
    dataset = VideoDataset(ROOT / config["data"]["val_manifest"], config["data"]["frames"],
                           config["data"]["height"], config["data"]["width"], config["data"]["fps"])
    rows = []
    for case, batch in enumerate(DataLoader(dataset, batch_size=1, num_workers=0)):
        if case >= args.cases:
            break
        pixels = batch["pixels"].to(device, dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full = pipeline.vae.encode_to_latent(pixels)
            # Match the actual AR block-end anchor positions; frame zero is a control.
            indices = [0] + list(range(pipeline.num_frame_per_block - 1,
                                       full.shape[1] - pipeline.num_frame_per_block,
                                       pipeline.num_frame_per_block))
            comparisons = []
            for index in indices:
                prefix = pipeline.vae.encode_to_latent(pixels[:, :, :4 * index + 1])[:, -1:]
                single = pipeline.vae.encode_to_latent(pixels[:, :, 4 * index:4 * index + 1])
                reference = full[:, index:index + 1]
                if prefix.shape != single.shape or prefix.shape != reference.shape:
                    raise RuntimeError(f"Latent shape mismatch at index {index}: {prefix.shape}, {single.shape}, {reference.shape}")
                comparisons.append({"latent_index": index, "pixel_index": 4 * index,
                                    "time_sec": 4 * index / config["data"]["fps"],
                                    "single_vs_full": compare(reference, single),
                                    "prefix_vs_full": compare(reference, prefix),
                                    "single_vs_prefix": compare(prefix, single)})
        rows.append({"source_id": batch["source_id"][0], "comparisons": comparisons})
        print(f"Aligned case {case + 1}/{min(args.cases, len(dataset))}: {batch['source_id'][0]}", flush=True)
    aggregate = {}
    for relation in ("single_vs_full", "prefix_vs_full", "single_vs_prefix"):
        metrics = [comparison[relation] for row in rows for comparison in row["comparisons"]
                   if comparison["latent_index"] > 0]
        aggregate[relation] = {key: torch.tensor([item[key] for item in metrics], dtype=torch.float64).mean(0).tolist()
                               for key in metrics[0]}
    result = {"cases": rows, "aggregate": aggregate, "num_cases": len(rows), "config": config,
              "aggregate_scope": "Mid-stream AR block-end anchors only; frame-zero control excluded",
              "note": "Full-video latents are diagnostics only; deployed RGB anchors use a single observed frame."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: {metric: value for metric, value in v.items() if 'channel' not in metric}
                      for k, v in aggregate.items()}, indent=2))


if __name__ == "__main__":
    main()
