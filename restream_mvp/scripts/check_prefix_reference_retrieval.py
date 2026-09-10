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

The query uses only causally visible pixel frames: ``L`` prefix latents expose
``4*(L-1)+1`` frames (``restream.reality_temporal``), so 6 latents give indices
0..20 and a 3-frame query is [0, 10, 20] - never 24 frames or frame 23.

Reported per unique target: pair accuracy, margins with target-level bootstrap
95% CI, AUROC, Recall@1/K in a per-target gallery, and a deterministic
derangement null test (query target != reference target and source-disjoint)
that replaces the earlier single cyclic shuffle. Both easy and hard comparisons
must have pair accuracy > 70% and a positive margin whose CI excludes zero.
"""
import argparse
import random
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
from restream.reality_temporal import (assert_visible_frame_indices, latent_prefix_boundary_index,
                                       pixel_count_for_latent_prefix)


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


def score_bank(query, bank):
    """Mean cosine between a pooled prefix query and a bank of pooled references."""
    return float(cosine(query[None], bank).mean())


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


def derangements(sources, samples, seed):
    """Deterministic permutations with query target != reference target and
    query source != reference source."""
    rng = random.Random(seed)
    count = len(sources)
    permutations, attempts = [], 0
    while len(permutations) < samples and attempts < samples * 200:
        attempts += 1
        order = list(range(count))
        rng.shuffle(order)
        if all(order[position] != position and sources[order[position]] != sources[position]
               for position in range(count)):
            permutations.append(order)
    if not permutations:
        raise ValueError("Could not build a source-disjoint derangement")
    return permutations


def similarity_columns(query_matrix, banks, key, indices):
    """Local query rows are already ordered by indices; banks use dataset IDs.

    matrix[j, i] = mean cosine between local query j and target indices[i].
    """
    if query_matrix.ndim != 2 or query_matrix.shape[0] != len(indices):
        raise ValueError("Expected one pooled query row per selected target")
    matrix = torch.zeros(len(indices), len(indices), dtype=torch.float64)
    for column, index in enumerate(indices):
        bank = banks[index][key]
        matrix[:, column] = cosine(query_matrix[:, None, :], bank[None, :, :]).mean(1)
    return matrix


def retrieval_gate(aggregate, accuracy_threshold=.70):
    if not 0 <= accuracy_threshold <= 1:
        raise ValueError("Accuracy threshold must be within [0,1]")
    gate = {"accuracy_threshold": accuracy_threshold}
    for kind in ("easy", "hard"):
        accuracy = aggregate[f"accuracy_{kind}"]
        margin = aggregate[f"margin_{kind}"]
        checks = {
            "pair_accuracy_pass": accuracy is not None and accuracy > accuracy_threshold,
            "margin_positive": margin["mean"] is not None and margin["mean"] > 0,
            "margin_ci_excludes_zero": margin["low"] is not None and margin["low"] > 0,
        }
        gate[kind] = {**checks, "pass": all(checks.values())}
    gate["pass"] = gate["easy"]["pass"] and gate["hard"]["pass"]
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_paired.yaml")
    parser.add_argument("--split", default="val")
    parser.add_argument("--cases", type=int, default=0, help="0 = every target; N = seeded sample of N targets")
    parser.add_argument("--target-seed", type=int, default=None)
    parser.add_argument("--prefix-frames", type=int, default=3, help="Visible prefix frames encoded per target (1-3)")
    parser.add_argument("--reference-count", type=int, default=0, help="References per set; 0 = paired reference_count")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--null-samples", type=int, default=1000, help="Deterministic derangements for the null test")
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
    visible_pixel_frames = pixel_count_for_latent_prefix(prefix_latents)
    frame_ids = assert_visible_frame_indices(prefix_frame_indices(visible_pixel_frames, args.prefix_frames),
                                             prefix_latents, "retrieval query frames")
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
    queries, banks, entries = {}, {}, []
    with torch.no_grad():
        for index in indices:
            row = rows[index]
            sample = dataset[index]
            pixels = sample["pixels"][:, frame_ids].permute(1, 0, 2, 3)  # (K,3,H,W), prefix frames only
            images = ((pixels.to(device).float() + 1) / 2).clamp(0, 1)
            features = encoder.encode_visual(images[None])[0]            # (K,P,D)
            query = pool_tokens(features).mean(0).cpu()                  # (P,D)
            queries[index] = query
            correct = row["reference_sets"][correct_kind][:reference_count]
            easy = row["reference_sets"]["wrong"][:reference_count]
            correct_mean = pool_tokens(dataset.reference_features(correct)).mean(0)
            hard_index = choose_hard_negative(index, rows, row_means, correct_mean, device)
            hard = rows[hard_index]["reference_sets"][correct_kind][:reference_count]
            banks[index] = {"correct": pool_tokens(dataset.reference_features(correct)),
                            "easy_wrong": pool_tokens(dataset.reference_features(easy)),
                            "hard_wrong": pool_tokens(dataset.reference_features(hard))}
            scores = {kind: score_bank(query, banks[index][kind]) for kind in banks[index]}
            gallery_scores, gallery_labels = [], []
            for reference, pooled in zip(correct, banks[index]["correct"]):
                gallery_scores.append(score_bank(query, pooled[None]))
                gallery_labels.append(1)
            for kind in ("easy_wrong", "hard_wrong"):
                for pooled in banks[index][kind]:
                    gallery_scores.append(score_bank(query, pooled[None]))
                    gallery_labels.append(0)
            entries.append({"index": index, "sample_id": row["sample_id"], "source_id": row["source_id"],
                            "hard_negative_sample_id": rows[hard_index]["sample_id"],
                            "hard_negative_source_id": rows[hard_index]["source_id"],
                            "prefix_frame_indices": frame_ids,
                            "prefix_frame_times": [float(sample["sampled_times"][i]) for i in frame_ids],
                            "visible_until": row.get("visible_until"),
                            "scores": scores,
                            "margin_easy": scores["correct"] - scores["easy_wrong"],
                            "margin_hard": scores["correct"] - scores["hard_wrong"],
                            "correct_gt_easy": scores["correct"] > scores["easy_wrong"],
                            "correct_gt_hard": scores["correct"] > scores["hard_wrong"],
                            "correct_gt_both": scores["correct"] > max(scores["easy_wrong"], scores["hard_wrong"]),
                            **gallery_recall(gallery_scores, gallery_labels, (1, 2, reference_count))})
    # Null test: many deterministic source-disjoint derangements. The query target
    # and the reference target always differ, so the accuracy should collapse to
    # chance if the prefix content itself carries the identity signal.
    query_matrix = torch.stack([queries[index] for index in indices])
    permutations = derangements([rows[index]["source_id"] for index in indices], args.null_samples, target_seed)
    columns = {key: similarity_columns(query_matrix, banks, key, indices)
               for key in ("correct", "easy_wrong", "hard_wrong")}
    null_easy, null_hard = [], []
    for permutation in permutations:
        null_easy.append(fraction_true(columns["correct"][query, position] > columns["easy_wrong"][query, position]
                                       for position, query in enumerate(permutation)))
        null_hard.append(fraction_true(columns["correct"][query, position] > columns["hard_wrong"][query, position]
                                       for position, query in enumerate(permutation)))
    pair_scores, pair_labels = [], []
    for entry in entries:
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
        "mean_correct_score": mean(entry["scores"]["correct"] for entry in entries),
        "mean_easy_score": mean(entry["scores"]["easy_wrong"] for entry in entries),
        "mean_hard_score": mean(entry["scores"]["hard_wrong"] for entry in entries),
    }
    null_test = {"permutations": len(permutations), "constraint": "query target != reference target and source-disjoint",
                 "accuracy_easy": bootstrap_ci(null_easy, min(args.bootstrap_samples, 2000), target_seed),
                 "accuracy_hard": bootstrap_ci(null_hard, min(args.bootstrap_samples, 2000), target_seed),
                 "null_accuracy_easy_mean": mean(null_easy), "null_accuracy_hard_mean": mean(null_hard)}
    null_test["signal_over_null_easy"] = (aggregate["accuracy_easy"] - null_test["null_accuracy_easy_mean"]
                                          if aggregate["accuracy_easy"] is not None else None)
    null_test["signal_over_null_hard"] = (aggregate["accuracy_hard"] - null_test["null_accuracy_hard_mean"]
                                          if aggregate["accuracy_hard"] is not None else None)
    gate = retrieval_gate(aggregate, args.accuracy_threshold)
    report = {"split": args.split, "correct_kind": correct_kind, "reference_count": reference_count,
              "prefix_frames": frame_ids, "prefix_latents": prefix_latents,
              "visible_pixel_frames": visible_pixel_frames,
              "prefix_latent_boundary_index": latent_prefix_boundary_index(prefix_latents),
              "temporal_mapping": "pixel_frames = 4*(prefix_latents-1)+1",
              "target_selection": {"mode": "all" if args.cases <= 0 or args.cases >= len(rows) else "seeded_sample",
                                   "count": len(indices), "seed": target_seed,
                                   "sample_ids": [rows[index]["sample_id"] for index in indices]},
              "encoder_identity": cache.identity,
              "train_manifest_sha256": manifest_digest(ROOT / config["data"]["train_manifest"]),
              "val_manifest_sha256": manifest_digest(ROOT / config["data"]["val_manifest"]),
              "selection_config_hash": selection_config_hash(config),
              "selection_protocol": memory["references"].get("selection_protocol", "offline_target_filtered"),
              "temporal_sampling": config["data"]["temporal_sampling"],
              "entries": entries, "aggregate": aggregate, "null_test": null_test, "gate": gate,
              "scope": "Frozen-DINO prefix retrieval only: no video model, no diffusion, no training. "
                       "The query uses only causally visible prefix frames. "
                       "Hard negatives are cross-source references chosen by similarity to the correct reference, not by the prefix query.",
              "notes": ["s_correct/s_easy/s_hard are mean cosine similarities between the pooled prefix query and the pooled reference features.",
                        "AUROC pools all (query, reference) pairs; accuracy and bootstrap CIs use unique targets.",
                        "The null test reuses other targets' prefix queries under many deterministic source-disjoint derangements."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"targets={aggregate['unique_targets']} frames={frame_ids} of 0..{latent_prefix_boundary_index(prefix_latents)} "
          f"accuracy_easy={aggregate['accuracy_easy']:.3f} accuracy_hard={aggregate['accuracy_hard']:.3f} "
          f"margin_easy={aggregate['margin_easy']['mean']:+.5f} CI=[{aggregate['margin_easy']['low']:+.5f}, {aggregate['margin_easy']['high']:+.5f}] "
          f"auroc={aggregate['auroc']:.3f} top1={aggregate['top1_correct']:.3f} "
          f"null_easy={null_test['null_accuracy_easy_mean']:.3f} null_hard={null_test['null_accuracy_hard_mean']:.3f} "
          f"gate={'PASS' if gate['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
