"""Fixed paired controls on train and held-out sources, for clean and mild histories."""
import torch
from .objective import future_loss
from .reality_data import write_json
from .reality_dataset import collate_reality
from .reality_metrics import memory_usage
from .reality_paired import (paired_references, paired_history, load_global_constant,
                             global_constant_provenance)
from .reality_runtime import PreserveHistory, prepare_reality

# Ordered control ladder used by every paired probe and by the summaries.
# base == none is an invariant; active_zero keeps the branch active with zero
# image content; global_constant is the fixed deduplicated train-pool mean.
CONTROL_KINDS = ("base", "none", "active_zero", "pair_mean", "global_constant", "correct", "wrong_source")


def _mean(entries, key):
    values = [entry[key] for entry in entries if entry.get(key) is not None]
    return sum(values) / len(values) if values else None


def aggregate_pairs(cases):
    """Aggregate one prefix mode; repeated noise seeds stay inside a target.

    Reported deltas are first computed per case (same target, same noise seed,
    same history) and then averaged over the noise seeds of each unique target,
    so every unique target counts once regardless of its number of seeds.
    """
    result = {}
    for mode in ("clean", "mild"):
        rows = [case for case in cases if case["prefix_mode"] == mode]
        variants = {}
        for kind in CONTROL_KINDS:
            entries = [case["variants"][kind] for case in rows if kind in case["variants"]]
            keys = sorted({key for entry in entries for key in entry})
            variants[kind] = {key: _mean(entries, key) for key in keys if any(entry.get(key) is not None for entry in entries)}
        unique = list(dict.fromkeys(row["sample_id"] for row in rows))
        unique_rows = [[row for row in rows if row["sample_id"] == sample_id] for sample_id in unique]

        def delta_means(minuend, subtrahend):
            """Loss(minuend) - Loss(subtrahend) averaged per unique target first.

            A positive value means the subtrahend variant is better (lower loss).
            """
            target_values = []
            for group in unique_rows:
                values = []
                for case in group:
                    first = case["variants"].get(minuend, {}).get("video_loss")
                    second = case["variants"].get(subtrahend, {}).get("video_loss")
                    if first is not None and second is not None:
                        values.append(first - second)
                if values:
                    target_values.append(sum(values) / len(values))
            if not target_values:
                return None, 0, 0
            return sum(target_values) / len(target_values), len(target_values), sum(v > 0 for v in target_values)

        core = {
            "U_correct": delta_means("none", "correct"),
            "G_branch": delta_means("none", "active_zero"),
            "G_generic": delta_means("active_zero", "global_constant"),
            "G_content": delta_means("global_constant", "correct"),
            "S_reference": delta_means("wrong_source", "correct"),
            "H_wrong": delta_means("wrong_source", "none"),
        }
        result[mode] = {"variants": variants, "target_noise_pairs": len(rows),
                        "unique_targets": len(unique),
                        "score_correct_gt_wrong": sum(row["variants"]["correct"]["relevance_score"] > row["variants"]["wrong_source"]["relevance_score"] for row in rows),
                        "score_correct_gt_wrong_unique_targets": sum(sum(row["variants"]["correct"]["relevance_score"] for row in group) / len(group) > sum(row["variants"]["wrong_source"]["relevance_score"] for row in group) / len(group) for group in unique_rows),
                        "loss_correct_lt_none_lt_wrong": sum(row["variants"]["correct"]["video_loss"] < row["variants"]["none"]["video_loss"] < row["variants"]["wrong_source"]["video_loss"] for row in rows),
                        "loss_correct_lt_none_lt_wrong_unique_targets": sum(sum(row["variants"]["correct"]["video_loss"] for row in group) / len(group) < sum(row["variants"]["none"]["video_loss"] for row in group) / len(group) < sum(row["variants"]["wrong_source"]["video_loss"] for row in group) / len(group) for group in unique_rows),
                        # Legacy aggregate deltas (equal weight over all pairs) kept for back-compat.
                        "correct_minus_none": variants["correct"]["video_loss"] - variants["none"]["video_loss"],
                        "correct_minus_global_constant": variants["correct"]["video_loss"] - variants["global_constant"]["video_loss"],
                        "global_constant_minus_none": variants["global_constant"]["video_loss"] - variants["none"]["video_loss"],
                        "wrong_minus_none": variants["wrong_source"]["video_loss"] - variants["none"]["video_loss"],
                        # Unique-target paired deltas: (mean, unique targets, targets > 0).
                        "core_deltas": {name: value[0] for name, value in core.items()},
                        "core_delta_targets": {name: value[1] for name, value in core.items()},
                        "core_delta_targets_positive": {name: value[2] for name, value in core.items()}}
    return result


@torch.no_grad()
def probe_paired(pipeline, memory, dataset, indices, config, device, output, split):
    cfg = config["reality_memory"]["objective"]["paired"]
    mode = memory.training
    memory.eval()
    cases = []
    global_mean = load_global_constant(config, dataset.cache, device)
    constant_record = global_constant_provenance(config, dataset.cache, device)
    try:
        for position, index in enumerate(indices):
            row = dataset.rows[index]
            correct, wrong = paired_references(row, cfg)
            gt, conditioning, anchor = prepare_reality(pipeline, collate_reality([dataset[index]]), device, config)
            context = conditioning["prompt_embeds"]
            empty = torch.empty(1, 0, dataset.cache.tokens, dataset.cache.channels, device=device)
            correct_features = dataset.reference_features(correct)
            wrong_features = dataset.reference_features(wrong)
            # Controls keep K and the trainable memory branch active while
            # removing scene-specific image content.
            zero_features = torch.zeros(cfg["reference_count"], dataset.cache.tokens, dataset.cache.channels, device=device)
            # Pair mean = current target's correct+wrong feature mean: a mixed
            # reference ablation, not a scene-independent constant.
            pair_mean_features = torch.cat((correct_features, wrong_features), 0).mean(0, keepdim=True).repeat(cfg["reference_count"], 1, 1)
            # Global constant = fixed mean over the deduplicated train-split
            # async+aligned pools (identity checked by load_global_constant).
            constant_features = global_mean.unsqueeze(0).repeat(cfg["reference_count"], 1, 1)
            references = {"base": None, "none": empty, "active_zero": zero_features[None],
                          "pair_mean": pair_mean_features[None].to(device),
                          "global_constant": constant_features[None],
                          "correct": correct_features[None].to(device),
                          "wrong_source": wrong_features[None].to(device)}
            for seed_offset in cfg["probe_noise_seeds"]:
                seed = config["seed"] + seed_offset + index + (0 if split == "train" else 10000)
                for prefix_mode in ("clean", "mild"):
                    history = paired_history(gt, anchor, cfg, device, seed + 30000, prefix_mode)
                    protected = cfg["degradation"]["protected_prefix"]
                    if not torch.equal(history[:, :protected], gt[:, :protected]):
                        raise RuntimeError("Mild prefix changed protected world-identity latents")
                    variants = {}
                    for kind, features in references.items():
                        cond, stats = conditioning, None
                        with torch.autocast(torch.device(device).type, enabled=torch.device(device).type == "cuda", dtype=torch.bfloat16):
                            if features is not None:
                                fused, stats = memory(context, features, torch.ones(features.shape[:2], device=device, dtype=torch.bool))
                                if kind == "none" and not torch.equal(fused, context):
                                    raise RuntimeError("No-memory context differs from base after training")
                                cond = {**conditioning, "prompt_embeds": fused}
                            loss = future_loss(pipeline, PreserveHistory(), gt, history, history[:, -1:], cond, anchor,
                                               torch.Generator(device=device).manual_seed(seed), 0)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite paired probe")
                        variants[kind] = {"video_loss": loss.item(), **(memory_usage(stats, kind == "wrong_source") if stats else {})}
                    if variants["base"]["video_loss"] != variants["none"]["video_loss"]:
                        raise RuntimeError("No-memory paired loss differs from base")
                    cases.append({"sample_id": row["sample_id"], "source_id": row["source_id"], "split": split,
                                  "correct_references": correct, "wrong_source_references": wrong,
                                  "prefix_mode": prefix_mode, "noise_seed": seed, "history_seed": seed + 30000,
                                  "protected_prefix_exact": True,
                                  "prefix_suffix_mse": (history[:, protected:].float() - gt[:, protected:anchor + 1].float()).square().mean().item(),
                                  "variants": variants})
            print(f"Paired probe {split} {position + 1}/{len(indices)}", flush=True)
        report = {"config": config, "split": split, "cases": cases, "aggregate": aggregate_pairs(cases),
                  "global_constant": constant_record,
                  "scope": "Fixed first-future-block teacher-forcing controls, not AR or long-horizon evidence. Noise seeds are repeated measures, not independent targets.",
                  "notes": ["Correct is same-source past-only proxy; wrong-source is not certified wrong-world.",
                            "Global constant repeats the deduplicated train-split async+aligned pool mean; its provenance (manifest/keys/selection digests) is checked at load and recorded in this report.",
                            "Active zero keeps K valid slots and the full trainable branch with raw DINO features set to zero; it isolates the learned context residual from any image statistics.",
                            "Pair mean is a mixed correct+wrong reference ablation, not a scene-independent constant.",
                            "Relevance is prompt/retrieved-feature cosine before the gate; ranking is directly supervised and is not by itself video efficacy.",
                            "Only the suffix of GT history is degraded; reference images never replace a latent."]}
        write_json(output, report)
        return report
    finally:
        memory.train(mode)