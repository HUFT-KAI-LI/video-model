"""Explicit MVP measurements; undefined perceptual/long-horizon claims remain null."""
import torch
from torch.nn import functional as F


def memory_usage(stats, wrong=False):
    active = stats["active"].bool()
    gate = stats["gate"][active].float().mean().item() if active.any() else None
    return {"memory_gate_mean": stats["gate"].float().mean().item(),
            "correct_memory_gate": gate if not wrong else None,
            "wrong_source_gate": gate if wrong else None,
            "relevance_score": stats["relevance_score"][active].float().mean().item() if active.any() else None,
            "raw_delta_norm": stats["raw_delta_norm"][active].float().mean().item() if active.any() else None,
            "applied_delta_norm": stats["applied_delta_norm"][active].float().mean().item() if active.any() else None,
            "memory_attention_entropy": stats["attention_entropy"].float().mean().item(),
            "memory_attention_entropy_normalized": stats["attention_entropy_normalized"].float().mean().item(),
            "valid_memory_tokens": stats["valid_memory_tokens"].float().mean().item()}


def visual_reference_metrics(generated_features, reference_features):
    if reference_features.shape[0] == 0:
        return {"reference_copy_score": None, "dino_world_similarity": None}
    generated = F.normalize(generated_features.float().mean(-2), dim=-1)
    references = F.normalize(reference_features.float().mean(-2), dim=-1)
    similarity = generated @ references.T
    return {"reference_copy_score": similarity.max().item(),
            "dino_world_similarity": similarity.max(-1).values.mean().item()}


def summarize_cases(cases):
    names = sorted({name for case in cases for name in case["variants"]})
    aggregate = {}
    for name in names:
        entries = [case["variants"][name] for case in cases if name in case["variants"]]
        aggregate[name] = {}
        for metric in ("future_latent_mse", "next_block_latent_mse", "memory_gate_mean", "correct_memory_gate",
                       "wrong_source_gate", "reference_copy_score", "dino_world_similarity", "memory_attention_entropy",
                       "memory_attention_entropy_normalized", "valid_memory_tokens", "relevance_score", "raw_delta_norm",
                       "applied_delta_norm", "pixel_motion_magnitude"):
            values = [entry[metric] for entry in entries if entry.get(metric) is not None]
            aggregate[name][metric] = sum(values) / len(values) if values else None
    gaps = {}
    for name, entry in aggregate.items():
        if name.startswith("async_k"):
            wrong = aggregate.get(name.replace("async_", "wrong_source_")) or aggregate.get(name.replace("async_", "wrong_"))
            if wrong:
                gaps[name] = wrong["future_latent_mse"] - entry["future_latent_mse"]
    return {"variants": aggregate, "correct_vs_wrong_gap": gaps, "base_quality": None}
