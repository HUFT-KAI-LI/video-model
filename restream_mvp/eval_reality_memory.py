"""Controlled R0 ablations. Same target, prefix, prompt and noise for every variant."""
import argparse
from pathlib import Path
import torch
from restream.anchor_injector import HardAnchorInjector
from restream.metrics import future_errors
from restream.reality_data import write_json
from restream.reality_dataset import collate_reality
from restream.reality_encoder import RealityEncoder
from restream.reality_metrics import memory_usage, visual_reference_metrics, summarize_cases
from restream.reality_runtime import (read_reality_config, make_memory, make_cache, make_dataset,
                                      prepare_reality, resume_signature)
from restream.runtime import ROOT, load_pipeline, rollout
from eval_reanchor import write_video


@torch.no_grad()
def evaluate(pipeline, memory, config, device, output, cases=None, counts=None, visual_metrics=False):
    count = config["eval"]["cases"] if cases is None else cases
    counts = sorted(set(config["eval"]["reference_counts"] if counts is None else counts))
    if count < 1 or not counts or min(counts) < 0:
        raise ValueError("Require positive case count and nonnegative reference counts")
    dataset = make_dataset(config, "val")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    mode = memory.training
    memory.eval()
    encoder = None
    if visual_metrics:
        cfg = config["reality_memory"]
        encoder = RealityEncoder(ROOT / cfg["encoder"]["path"], cfg["projector"]["memory_dim"],
                                 cfg["projector"]["num_memory_tokens"], cfg["encoder"]["image_size"]).to(device).eval()
    results = []
    try:
        for case in range(min(count, len(dataset))):
            batch = collate_reality([dataset[case]])
            row = dataset.rows[case]
            gt, conditioning, anchor = prepare_reality(pipeline, batch, device, config)
            prefix = gt[:, :anchor + 1]
            real = HardAnchorInjector(pipeline.vae).encode_anchor(batch["pixels"][:, :, 4 * anchor].to(device, dtype=gt.dtype))
            seed = config["seed"] + case
            noise = torch.randn(gt[:, anchor + 1:].shape, generator=torch.Generator(device=device).manual_seed(seed + 10000),
                                device=device, dtype=gt.dtype)
            empty = torch.empty(1, 0, dataset.cache.tokens, dataset.cache.channels, device=device)
            variants = [("base", None, prefix, False), ("no_memory", empty, prefix, False),
                        ("hard_anchor", None, torch.cat((prefix[:, :-1], real.to(gt.dtype)), 1), False),
                        ("oracle_gt_state", None, prefix, False)]
            # In this new continuation task the prefix is clean GT, so Oracle == Base.
            # The older corrupted-history recovery diagnostic remains in eval_reanchor.py.
            for k in counts:
                if k == 0:
                    continue
                for kind in ("aligned", "async", "wrong", "shuffled"):
                    pool = row["reference_sets"].get(kind)
                    if kind == "shuffled":
                        donor = next(item for item in dataset.rows[case + 1:] + dataset.rows[:case]
                                     if item["source_id"] != row["source_id"])
                        pool = donor["reference_sets"]["async"]
                    if len(pool) < k:
                        raise ValueError(f"Only {len(pool)} cached {kind} references for K={k}; rebuild pools/cache")
                    features = dataset.reference_features(pool[:k])[None].to(device)
                    variants.append((f"{kind}_k{k}", features, prefix, kind in ("wrong", "shuffled")))
            record = {"sample_id": row["sample_id"], "source_id": row["source_id"], "seed": seed,
                      "anchor_latent_index": anchor, "target_start": row["target_start"], "variants": {}}
            folder = output / f"case_{case:03d}"
            folder.mkdir(exist_ok=True)
            base_generated = None
            for name, features, history, wrong in variants:
                cond = conditioning
                scores = {"base_quality": None, "reference_copy_score": None, "dino_world_similarity": None,
                          "memory_gate_mean": None, "correct_memory_gate": None, "wrong_memory_gate": None,
                          "memory_attention_entropy": None}
                if features is not None:
                    mask = torch.ones(features.shape[:2], device=device, dtype=torch.bool)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        fused, stats = memory(conditioning["prompt_embeds"], features, mask)
                    if name == "no_memory" and not torch.equal(fused, conditioning["prompt_embeds"]):
                        raise RuntimeError("No-memory conditioning differs from base")
                    cond = {**conditioning, "prompt_embeds": fused}
                    scores.update(memory_usage(stats, wrong))
                generated = rollout(pipeline, history, cond, noise.clone(), torch.Generator(device=device).manual_seed(seed + 20000))
                if not torch.isfinite(generated).all():
                    raise RuntimeError(f"Nonfinite rollout: {name}")
                if name == "base":
                    base_generated = generated.clone()
                if name == "no_memory":
                    scores["max_abs_difference_from_base"] = (generated - base_generated).abs().max().item()
                    if not torch.equal(generated, base_generated):
                        raise RuntimeError("No-memory rollout differs from base under identical noise")
                scores.update(future_errors(generated, gt, anchor, row["target_sec"]))
                end = anchor + 1 + pipeline.num_frame_per_block
                scores["next_block_latent_mse"] = (generated[:, anchor + 1:end].float() - gt[:, anchor + 1:end].float()).square().mean().item()
                decoded = pipeline.vae.decode_to_pixel(generated)[0].float()
                future_pixels = decoded[4 * anchor + 1:]
                scores["pixel_motion_magnitude"] = (future_pixels[1:] - future_pixels[:-1]).abs().mean().item()
                if encoder is not None and features is not None and features.shape[1]:
                    # Decoder uses T,C,H,W; RealityEncoder uses B,K,C,H,W.
                    visual = encoder.encode_visual(((future_pixels[::4] + 1) / 2).clamp(0, 1)[None])[0]
                    scores.update(visual_reference_metrics(visual, features[0]))
                frames = ((decoded.clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
                write_video(folder / f"{name}.mp4", frames, config["data"]["fps"], [name], -100)
                record["variants"][name] = scores
                print(f"case={case} variant={name} future_mse={scores['future_latent_mse']:.6f}", flush=True)
            record["correct_vs_wrong_gap"] = {f"k{k}": record["variants"][f"wrong_k{k}"]["future_latent_mse"] - record["variants"][f"async_k{k}"]["future_latent_mse"] for k in counts if k}
            write_json(folder / "metrics.json", record)
            results.append(record)
        summary = {"config": config, "cases": results, "aggregate": summarize_cases(results),
                   "visual_metrics": "sample every fourth future pixel frame" if encoder else "not requested; null",
                   "notes": ["R0 uses prompt context, not state-conditioned retrieval.",
                             "Oracle equals Base here because continuation starts from clean GT; use eval_reanchor.py for corrupted-history Oracle recovery.",
                             "Shuffled memory swaps reference sets across sources; permuting order inside a set should not change this model.",
                             "Histogram scene filtering and same-source sampling are only asynchronous-world proxies."]}
        write_json(output / "metrics.json", summary)
        return summary
    finally:
        memory.train(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cases", type=int)
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--visual-metrics", action="store_true")
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reality_memory/evaluation")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("R0 evaluation starts after review; pass --reviewed")
    config = read_reality_config(args.config)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda")
    pipeline = load_pipeline(config, device)
    memory = make_memory(config).to(device)
    if args.checkpoint:
        path = args.checkpoint / "state.pt" if args.checkpoint.is_dir() else args.checkpoint
        saved = torch.load(path, map_location="cpu", weights_only=False)
        expected = resume_signature(config, make_cache(config))
        if saved.get("stage") != "r0" or any(saved["signature"][key] != value for key, value in expected.items()):
            raise ValueError("Checkpoint incompatible with model, encoder or manifests")
        memory.load_state_dict(saved["memory"], strict=True)
    evaluate(pipeline, memory, config, device, args.output, args.cases, args.counts, args.visual_metrics)


if __name__ == "__main__":
    main()
