"""Fixed paired controls on train and held-out sources, for clean and mild histories."""
import torch
from .objective import future_loss
from .reality_data import write_json
from .reality_dataset import collate_reality
from .reality_metrics import memory_usage
from .reality_paired import paired_references, paired_history
from .reality_runtime import PreserveHistory, prepare_reality


def aggregate_pairs(cases):
    result = {}
    for mode in ("clean", "mild"):
        rows = [case for case in cases if case["prefix_mode"] == mode]
        variants = {}
        for kind in ("base", "none", "constant", "correct", "wrong_source"):
            entries = [case["variants"][kind] for case in rows]
            variants[kind] = {key: sum(e[key] for e in entries if e[key] is not None) / sum(e[key] is not None for e in entries)
                              for key in entries[0] if any(e[key] is not None for e in entries)}
        unique = list(dict.fromkeys(r["sample_id"] for r in rows))
        unique_rows = [[r for r in rows if r["sample_id"] == sample_id] for sample_id in unique]
        result[mode] = {"variants": variants, "target_noise_pairs": len(rows),
                        "unique_targets": len(unique),
                        "score_correct_gt_wrong": sum(r["variants"]["correct"]["relevance_score"] > r["variants"]["wrong_source"]["relevance_score"] for r in rows),
                        "score_correct_gt_wrong_unique_targets": sum(sum(r["variants"]["correct"]["relevance_score"] for r in group) / len(group) > sum(r["variants"]["wrong_source"]["relevance_score"] for r in group) / len(group) for group in unique_rows),
                        "loss_correct_lt_none_lt_wrong": sum(r["variants"]["correct"]["video_loss"] < r["variants"]["none"]["video_loss"] < r["variants"]["wrong_source"]["video_loss"] for r in rows),
                        "loss_correct_lt_none_lt_wrong_unique_targets": sum(sum(r["variants"]["correct"]["video_loss"] for r in group) / len(group) < sum(r["variants"]["none"]["video_loss"] for r in group) / len(group) < sum(r["variants"]["wrong_source"]["video_loss"] for r in group) / len(group) for group in unique_rows),
                        "correct_minus_none": variants["correct"]["video_loss"] - variants["none"]["video_loss"],
                        "correct_minus_constant": variants["correct"]["video_loss"] - variants["constant"]["video_loss"],
                        "constant_minus_none": variants["constant"]["video_loss"] - variants["none"]["video_loss"],
                        "wrong_minus_none": variants["wrong_source"]["video_loss"] - variants["none"]["video_loss"]}
    return result


@torch.no_grad()
def probe_paired(pipeline, memory, dataset, indices, config, device, output, split):
    cfg = config["reality_memory"]["objective"]["paired"]
    mode = memory.training
    memory.eval()
    cases = []
    try:
        for position, index in enumerate(indices):
            row = dataset.rows[index]
            correct, wrong = paired_references(row, cfg)
            gt, conditioning, anchor = prepare_reality(pipeline, collate_reality([dataset[index]]), device, config)
            context = conditioning["prompt_embeds"]
            empty = torch.empty(1, 0, dataset.cache.tokens, dataset.cache.channels, device=device)
            correct_features = dataset.reference_features(correct)
            wrong_features = dataset.reference_features(wrong)
            # Constant control keeps K and the trainable memory branch active while
            # removing scene-specific image content. It is computed only from the
            # current split's pair pool, never from the target's future pixels.
            constant_features = torch.cat((correct_features, wrong_features), 0).mean(0, keepdim=True).repeat(cfg["reference_count"], 1, 1)
            references = {"base": None, "none": empty, "constant": constant_features[None].to(device),
                          "correct": correct_features[None].to(device), "wrong_source": wrong_features[None].to(device)}
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
                  "scope": "Fixed first-future-block teacher-forcing controls, not AR or long-horizon evidence. Noise seeds are repeated measures, not independent targets.",
                  "notes": ["Correct is same-source past-only proxy; wrong-source is not certified wrong-world.",
                            "Constant repeats the mean feature over this target's paired pool; it controls active branch and K, but is not an independently trained constant-memory baseline.",
                            "Relevance is prompt/retrieved-feature cosine before the gate; ranking is directly supervised and is not by itself video efficacy.",
                            "Only the suffix of GT history is degraded; reference images never replace a latent."]}
        write_json(output, report)
        return report
    finally:
        memory.train(mode)
