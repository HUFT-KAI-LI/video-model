"""Comparable fixed-noise probes and grouped gradients for short R0 experiments."""
import torch
from .objective import future_loss
from .reality_data import write_json
from .reality_dataset import collate_reality
from .reality_metrics import memory_usage
from .reality_runtime import prepare_reality, PreserveHistory


def gradient_norms(memory):
    groups = {}
    for name in ("output", "projector", "query", "key", "value", "gate"):
        grads = [p.grad for p in getattr(memory, name).parameters() if p.grad is not None]
        groups[name] = sum(g.detach().float().square().sum() for g in grads).sqrt().item() if grads else None
    return groups


@torch.no_grad()
def probe_subset(pipeline, memory, dataset, indices, config, device, output, count=2):
    """Compare each reference type on identical target/noise before and after overfit.

    Teacher forcing is a learning diagnostic, not an autoregressive quality metric.
    Dropout and regularizers are disabled. No timestamps/labels enter the memory model.
    """
    mode = memory.training
    memory.eval()
    cases = []
    try:
        for position, index in enumerate(indices):
            row = dataset.rows[index]
            batch = collate_reality([dataset[index]])
            gt, conditioning, anchor = prepare_reality(pipeline, batch, device, config)
            variants = {}
            for kind in ("base", "none", "async", "aligned", "wrong_source"):
                cond, stats = conditioning, None
                if kind != "base":
                    pool = [] if kind == "none" else row["reference_sets"]["wrong" if kind == "wrong_source" else kind][:count]
                    features = dataset.reference_features(pool)[None].to(device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        fused, stats = memory(cond["prompt_embeds"], features,
                                              torch.ones(features.shape[:2], device=device, dtype=torch.bool))
                    if kind == "none" and not torch.equal(fused, cond["prompt_embeds"]):
                        raise RuntimeError("No-memory probe changed base context")
                    cond = {**cond, "prompt_embeds": fused}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = future_loss(pipeline, PreserveHistory(), gt, gt[:, :anchor + 1], gt[:, anchor:anchor + 1],
                                       cond, anchor, torch.Generator(device=device).manual_seed(config["seed"] + index + 50000), 0)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite fixed-noise probe loss")
                variants[kind] = {"video_loss": loss.item(), **(memory_usage(stats, kind == "wrong_source") if stats else {})}
            if variants["none"]["video_loss"] != variants["base"]["video_loss"]:
                raise RuntimeError("No-memory teacher-forcing differs from base")
            cases.append({"sample_id": row["sample_id"], "training_reference_kind": row["reference_kind"],
                          "noise_seed": config["seed"] + index + 50000, "variants": variants})
            print(f"Probe {position + 1}/{len(indices)}: async={variants['async']['video_loss']:.6f}, wrong_source={variants['wrong_source']['video_loss']:.6f}", flush=True)
        aggregate = {}
        for kind in cases[0]["variants"]:
            entries = [case["variants"][kind] for case in cases]
            aggregate[kind] = {key: sum(e[key] for e in entries if e[key] is not None) / sum(e[key] is not None for e in entries)
                               for key in entries[0] if any(e[key] is not None for e in entries)}
        result = {"config": config, "cases": cases, "aggregate": aggregate, "references_per_variant": count,
                  "scope": "fixed-noise first-future-block teacher-forcing; clean GT prefix; same-source past-frame proxy",
                  "wrong_source_note": "Different source is not necessarily irrelevant world evidence",
                  "correct_vs_wrong_video_loss_gap": aggregate["wrong_source"]["video_loss"] - aggregate["async"]["video_loss"]}
        write_json(output, result)
        return result
    finally:
        memory.train(mode)
