"""Audit and summarize completed paired probes without running a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json


def read(path):
    return json.loads(path.read_text())


def comparison(before, after):
    def identity(case):
        return {key: case[key] for key in ("sample_id", "source_id", "prefix_mode", "noise_seed", "history_seed",
                                           "correct_references", "wrong_source_references")}
    if [identity(c) for c in before["cases"]] != [identity(c) for c in after["cases"]]:
        raise ValueError("Before/after controls changed targets/references/history/noise")
    for first, last in zip(before["cases"], after["cases"]):
        if first["variants"]["base"]["video_loss"] != last["variants"]["base"]["video_loss"]:
            raise ValueError("Frozen Base changed between probes")
        if last["variants"]["base"]["video_loss"] != last["variants"]["none"]["video_loss"]:
            raise ValueError("No-memory invariant failed")
    results = {}
    for mode in ("clean", "mild"):
        a, b = before["aggregate"][mode], after["aggregate"][mode]
        loss = {kind: values["video_loss"] for kind, values in b["variants"].items()}
        scores = {kind: b["variants"][kind]["relevance_score"] for kind in ("correct", "wrong_source")}
        rows = [c for c in after["cases"] if c["prefix_mode"] == mode]
        target_means = []
        for sample_id in dict.fromkeys(c["sample_id"] for c in rows):
            selected = [c for c in rows if c["sample_id"] == sample_id]
            means = {kind: statistics.mean(c["variants"][kind]["video_loss"] for c in selected)
                     for kind in ("correct", "none", "wrong_source")}
            target_means.append({"sample_id": sample_id, **means,
                                 "ordered": means["correct"] < means["none"] < means["wrong_source"]})
        results[mode] = {"loss_before": {kind: values["video_loss"] for kind, values in a["variants"].items()},
                         "loss_after": loss, "relevance_before": {k: a["variants"][k]["relevance_score"] for k in scores},
                         "relevance_after": scores,
                         "gate_before": {k: a["variants"][k]["memory_gate_mean"] for k in scores},
                         "gate_after": {k: b["variants"][k]["memory_gate_mean"] for k in scores},
                         "correct_change_vs_none_percent": (loss["correct"] / loss["none"] - 1) * 100,
                         "wrong_change_vs_none_percent": (loss["wrong_source"] / loss["none"] - 1) * 100,
                         "mean_loss_ordered": loss["correct"] < loss["none"] < loss["wrong_source"],
                         "score_ordered_pairs": b["score_correct_gt_wrong"],
                         "loss_ordered_pairs": b["loss_correct_lt_none_lt_wrong"],
                         "target_noise_pairs": len(rows), "target_means": target_means,
                         "loss_ordered_targets": sum(t["ordered"] for t in target_means)}
    clean = results["clean"]["loss_after"]["none"]
    results["degradation_effect_on_none_percent"] = (results["mild"]["loss_after"]["none"] / clean - 1) * 100
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "checkpoints/reality_memory_paired_review")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/paired_review")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run / "train_rank_0.jsonl").read_text().splitlines()]
    step, updates = rows[-1]["batch_step"], rows[-1]["optimizer_step"]
    if [r["batch_step"] for r in rows] != list(range(1, step + 1)) or sum(r["optimizer_updated"] for r in rows) != updates:
        raise ValueError("Batch/update accounting inconsistent")
    checkpoint = args.run / f"step_{step:04d}/state.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["batch_step"] != step or state["optimizer_step"] != updates or state["budget"]["max_updates"] != updates:
        raise ValueError("Checkpoint did not reach the requested effective update budget")
    if {int(item["step"]) for item in state["optimizer"]["state"].values()} != {updates}:
        raise ValueError("Actual AdamW state step differs from the logged effective updates")
    comparisons, sources = {}, {}
    for split in ("train", "val"):
        before = args.run / f"paired_before_step_0000_{split}_rank_0.json"
        after = args.run / f"paired_after_step_{step:04d}_{split}_rank_0.json"
        first, last = read(before), read(after)
        comparisons[split] = comparison(first, last)
        sources[split] = {c["source_id"] for c in last["cases"]} | {
            r["source_id"] for c in last["cases"] for key in ("correct_references", "wrong_source_references") for r in c[key]}
        for path, label in ((before, "before"), (after, "after")):
            shutil.copyfile(path, args.output / f"{split}_{label}.json")
    if sources["train"] & sources["val"]:
        raise ValueError("Train/val target or reference source overlap")
    shutil.copyfile(args.run / "train_rank_0.jsonl", args.output / "train_steps.jsonl")
    report = {"batch_step": step, "optimizer_step": updates, "optimizer_state_step_verified": True,
              "sample_visits": dict(Counter(r["sample_id"] for r in rows)), "config": state["config"],
              "comparisons": comparisons, "train_val_sources_disjoint": True,
              "performance": {"sum_training_step_seconds": sum(r["step_time"] for r in rows),
                              "median_step_seconds": statistics.median(r["step_time"] for r in rows),
                              "peak_vram_bytes": max(r["peak_vram_bytes"] for r in rows)},
              "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
              "scope": "One training initialization, 16 training and 4 held-out targets, two noise seeds per target, first-future-block teacher forcing. Supervised relevance ordering alone is not evidence of video utility."}
    write_json(args.output / "summary.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    labels = [f"{split}/{mode}" for split in ("train", "val") for mode in ("clean", "mild")]
    for offset, kind, color in ((-.17, "correct", "#347ba5"), (.17, "wrong", "#be7844")):
        values = [comparisons[s][m][f"{kind}_change_vs_none_percent"] for s in ("train", "val") for m in ("clean", "mild")]
        axes[0].bar([i + offset for i in range(4)], values, .34, label=kind, color=color)
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylabel("Video loss vs No Memory (%)")
    axes[0].set_title("Paired fixed-noise controls; lower is better")
    axes[0].legend()
    for split in ("train", "val"):
        values = [comparisons[split]["mild"][f"relevance_{phase}"]["correct"] - comparisons[split]["mild"][f"relevance_{phase}"]["wrong_source"] for phase in ("before", "after")]
        axes[1].plot(["Before", "After"], values, marker="o", label=split)
    axes[1].axhline(0, color="black", linewidth=.7)
    axes[1].set_ylabel("Correct minus wrong-source relevance")
    axes[1].set_title("Directly supervised cosine score")
    axes[1].legend()
    fig.savefig(args.output / "paired_controls.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"updates": updates, "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
