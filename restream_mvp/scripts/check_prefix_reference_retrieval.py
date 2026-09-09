"""Prefix-state retrieval gate: can the visible video prefix identify the
same-world reference set?

No video-model training, no diffusion, no optimizer: this script only encodes
the visible prefix with the same frozen DINOv2 used for the cached references and
measures retrieval against three galleries:

  * correct      -- the target's own same-source past references;
  * easy wrong   -- the manifest's random donor-source references;
  * hard wrong   -- a cross-source reference chosen by DINO similarity to the
                    correct reference (never by the prefix query, so hard-negative
                    selection cannot leak into the score).

Reported per unique target: pair accuracy, margins with target-level bootstrap
95% CI, AUROC over query-reference pairs, Recall@1/K in a per-target gallery, and
a shuffled-query control that scores another target's prefix query against this
target's references (chance level). The gate passes when pair accuracy > 70% and
the correct-minus-wrong margin is positive with a CI that excludes zero.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json
from restream.reality_encoder import RealityEncoder
from restream.reality_runtime import read_reality_config, make_cache, make_dataset
from restream.reality_selection import manifest_digest, select_targets, selection_config_hash
from restream.reality_stats import auroc, bootstrap_ci, fraction_true, mean


def pool_tokens(features):
    """(..., P, D) -> (..., D) by averaging memory tokens."""
    return features.float().mean(-2)


def cosine(first, second):
    return torch.nn.functional.cosine_similarity(first.float(), second.float(), dim=-1)


def prefix_frame_indices(prefix_frames, count):
    """Visible-prefix frame indices; a single frame is the last visible one."""
    if count < 1 or count > prefix_frames:
        raise ValueError(f"prefix frames must be within 1..{prefix_frames}")
    if count == 1:
        return [prefix_frames - 1]
    return sorted(set(np.linspace(0, prefix_frames - 1, count).round().astype(int).tolist()))


def choose_hard_negative(index, rows, row_means, correct_mean, device):
    """Most DINO-similar cross-source row, selected from the correct reference."""
    similarity = cosine(correct_mean[None].to(device), row_means.to(device)).cpu()
    source = rows[index]["source_id"]
    for position, row in enumerate(rows):
        if position == index or row["source_id"] == source:
            similarity[position] = -torch.inf
    if not torch.isfinite(similarity).any():
        raise ValueError("No cross-source candidate for a hard negative")
    return int(similarity.argmax())


def score_set(query, references, dataset, device):
    """Mean cosine between a pooled prefix query and pooled reference features."""
    if not references:
        return None
    features = pool_tokens(dataset.reference_features(references).to(device))
    return float(cosine(query[None], features).mean())


def gallery_recall(scores, labels, ks):
    """Recall@k over one target's gallery; labels 1 = correct."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    positives = sum(labels)
    if not positives:
        return {f"recall@{k}": None for k in ks} | {"top1_correct": None}
    result = {}
    for k in ks:
        result[f"recall@{k}"] = sum(labels[i] for i in order[:k]) / positives
    result["top1_correct"] = bool(labels[order[0]])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_paired.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cases", type=int, default=0, help="0 = every target; N = seeded sample of N targets")
    parser.add_argument("--target-seed", type=int, default=None)
    parser.add_argument("--prefix-frames", type=int, default=3, help="Visible prefix frames encoded per target (1-3)")
    parser.add_argument("--reference-count", type=int, default=0, help="References per set; 0 = paired reference_count")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--accuracy-threshold", type=float, default=.70)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/prefix_retrieval/summary.json")
    args = parser.parse_args()
    config = read_reality_config(args.config)
    memory = config["reality_memory"]
    paired = memory["objective"]["paired"]
    correct_kind = paired["correct_kind"]
    reference_count = args.reference_count or paired["reference_count"]
    prefix_latents = memory["objective"]["prefix_latents"]
    target_seed = config["seed"] if args.target_seed is None else args.target_seed
    device = torch.device(args.device)
    cache = make_cache(config)
    dataset = make_dataset(config, args.split, cache)
    encoder = RealityEncoder(ROOT / memory["encoder"]["path"], memory["projector"]["memory_dim"],
                             memory["projector"]["num_memory_tokens"], memory["encoder"]["image_size"]).to(device).eval()
    if encoder.identity != cache.identity:
        raise ValueError("Prefix encoder identity differs from the reference feature cache")
    indices = select_targets(dataset.rows, args.cases, target_seed)
    prefix_frames = 4 * prefix_latents
    frame_ids = prefix_frame_indices(prefix_frames, args.prefix_frames)
    rows = dataset.rows
    # Cross-source gallery means for hard negatives, computed from the same kind
    # as the correct references and independent of any prefix query.
    row_means = []
    for row in rows:
        pool = row["reference_sets"][correct_kind]
        if len(pool) < reference_count:
            raise ValueError(f"Row {row['sample_id']} has only {len(pool)} {correct_kind} references")
        row_means.append(pool_tokens(dataset.reference_features(pool[:reference_count])).mean(0))
    row_means = torch.stack(row_means)
    queries, entries = {}, []
    with torch.no_grad():
        for index in indices:
            row = rows[index]
            sample = dataset[index]
            pixels = sample["pixels"][:, frame_ids].permute(1, 0, 2, 3)  # (K,3,H,W)
            images = ((pixels.to(device).float() + 1) / 2).clamp(0, 1)
            features = encoder.encode_visual(images[None])[0]            # (K,P,D)
            query = pool_tokens(features).mean(0)                        # (P,D)
            queries[index] = query
            correct = row["reference_sets"][correct_kind][:reference_count]
            easy = row["reference_sets"]["wrong"][:reference_count]
            correct_mean = pool_tokens(dataset.reference_features(correct)).mean(0)
            hard_index = choose_hard_negative(index, rows, row_means, correct_mean, device)
            hard = rows[hard_index]["reference_sets"][correct_kind][:reference_count]
            scores = {"correct": score_set(query, correct, dataset, device),
                      "easy_wrong": score_set(query, easy, dataset, device),
                      "hard_wrong": score_set(query, hard, dataset, device)}
            gallery_scores, gallery_labels = [], []
            for reference in correct:
                gallery_scores.append(score_set(query, [reference], dataset, device))
                gallery_labels.append(1)
            for reference in list(easy) + list(hard):
                gallery_scores.append(score_set(query, [reference], dataset, device))
                gallery_labels.append(0)
            entries.append({"index": index, "sample_id": row["sample_id"], "source_id": row["source_id"],
                            "hard_negative_sample_id": rows[hard_index]["sample_id"],
                            "hard_negative_source_id": rows[hard_index]["source_id"],
                            "prefix_frame_indices": frame_ids,
                            "prefix_frame_times": [float(sample["sampled_times"][i]) for i in frame_ids],
                            "scores": scores,
                            "margin_easy": scores["correct"] - scores["easy_wrong"],
                            "margin_hard": scores["correct"] - scores["hard_wrong"],
                            "correct_gt_easy": scores["correct"] > scores["easy_wrong"],
                            "correct_gt_hard": scores["correct"] > scores["hard_wrong"],
                            "correct_gt_both": scores["correct"] > max(scores["easy_wrong"], scores["hard_wrong"]),
                            **gallery_recall(gallery_scores, gallery_labels, (1, 2, reference_count))})
    # Shuffled-query control: score another target's prefix query against this
    # target's references; should be near chance.
    control = []
    for position, index in enumerate(indices):
        other = indices[(position + 1) % len(indices)]
        row = rows[index]
        correct = row["reference_sets"][correct_kind][:reference_count]
        easy = row["reference_sets"]["wrong"][:reference_count]
        with torch.no_grad():
            control.append(score_set(queries[other], correct, dataset, device) > score_set(queries[other], easy, dataset, device))
    pair_scores, pair_labels = [], []
    for entry, index in zip(entries, indices):
        for label, key in ((1, "correct"), (0, "easy_wrong"), (0, "hard_wrong")):
            pair_scores.append(entry["scores"][key])
            pair_labels.append(label)
    aggregate = {
        "unique_targets": len(entries),
        "accuracy_easy": fraction_true(entry["correct_gt_easy"] for entry in entries),
        "accuracy_hard": fraction_true(entry["correct_gt_hard"] for entry in entries),
        "accuracy_both": fraction_true(entry["correct_gt_both"] for entry in entries),
        "margin_easy": bootstrap_ci([entry["margin_easy"] for entry in entries], args.bootstrap_samples, target_seed),
        "margin_hard": bootstrap_ci([entry["margin_hard"] for entry in entries], args.bootstrap_samples, target_seed),
        "auroc": auroc(pair_scores, pair_labels),
        "recall@1": mean(entry["recall@1"] for entry in entries),
        "recall@2": mean(entry["recall@2"] for entry in entries),
        f"recall@{reference_count}": mean(entry[f"recall@{reference_count}"] for entry in entries),
        "top1_correct": fraction_true(entry["top1_correct"] for entry in entries),
        "shuffled_query_accuracy": fraction_true(control),
        "mean_correct_score": mean(entry["scores"]["correct"] for entry in entries),
        "mean_easy_score": mean(entry["scores"]["easy_wrong"] for entry in entries),
        "mean_hard_score": mean(entry["scores"]["hard_wrong"] for entry in entries),
    }
    gate = {"accuracy_threshold": args.accuracy_threshold,
            "pair_accuracy_pass": aggregate["accuracy_easy"] is not None and aggregate["accuracy_easy"] > args.accuracy_threshold,
            "margin_positive": aggregate["margin_easy"]["mean"] is not None and aggregate["margin_easy"]["mean"] > 0,
            "margin_ci_excludes_zero": aggregate["margin_easy"]["low"] is not None and aggregate["margin_easy"]["low"] > 0}
    gate["pass"] = all(gate[key] for key in ("pair_accuracy_pass", "margin_positive", "margin_ci_excludes_zero"))
    report = {"split": args.split, "correct_kind": correct_kind, "reference_count": reference_count,
              "prefix_frames": frame_ids, "prefix_latents": prefix_latents,
              "target_selection": {"mode": "all" if args.cases <= 0 or args.cases >= len(rows) else "seeded_sample",
                                   "count": len(indices), "seed": target_seed,
                                   "sample_ids": [rows[index]["sample_id"] for index in indices]},
              "encoder_identity": cache.identity,
              "train_manifest_sha256": manifest_digest(ROOT / config["data"]["train_manifest"]),
              "selection_config_hash": selection_config_hash(config),
              "entries": entries, "aggregate": aggregate, "gate": gate,
              "scope": "Frozen-DINO prefix retrieval only: no video model, no diffusion, no training. "
                       "Hard negatives are cross-source references chosen by similarity to the correct reference, not by the prefix query.",
              "notes": ["s_correct/s_easy/s_hard are mean cosine similarities between the pooled prefix query and the pooled reference features.",
                        "AUROC pools all (query, reference) pairs; accuracy and bootstrap CIs use unique targets.",
                        "The shuffled-query control reuses another target's prefix query to estimate chance accuracy."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"targets={aggregate['unique_targets']} accuracy_easy={aggregate['accuracy_easy']:.3f} "
          f"accuracy_hard={aggregate['accuracy_hard']:.3f} margin_easy={aggregate['margin_easy']['mean']:+.5f} "
          f"CI=[{aggregate['margin_easy']['low']:+.5f}, {aggregate['margin_easy']['high']:+.5f}] "
          f"auroc={aggregate['auroc']:.3f} recall@1={aggregate['recall@1']:.3f} "
          f"shuffled={aggregate['shuffled_query_accuracy']:.3f} gate={'PASS' if gate['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
