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

CONTROL_KINDS = ("base", "none", "active_zero", "pair_mean", "global_constant", "correct", "wrong_source")
CORE_DELTAS = {"U_correct": ("none", "correct"), "G_branch": ("none", "active_zero"),
               "G_generic": ("active_zero", "global_constant"), "G_content": ("global_constant", "correct"),
               "S_reference": ("wrong_source", "correct")}


def read(path):
    return json.loads(path.read_text())


def target_level_deltas(rows):
    """Paired deltas averaged over a target's noise seeds, one entry per unique target."""
    entries = []
    for sample_id in dict.fromkeys(c["sample_id"] for c in rows):
        selected = [c for c in rows if c["sample_id"] == sample_id]
        means = {}
        for kind in CONTROL_KINDS:
            values = [c["variants"][kind]["video_loss"] for c in selected if kind in c["variants"]]
            if values:
                means[kind] = statistics.mean(values)
        deltas = {name: (means[a] - means[b]) if a in means and b in means else None
                  for name, (a, b) in CORE_DELTAS.items()}
        entries.append({"sample_id": sample_id, **means, "core_deltas": deltas,
                        "ordered": all(kind in means for kind in ("correct", "none", "wrong_source"))
                        and means["correct"] < means["none"] < means["wrong_source"]})
    return entries


def asset_consistency(comparisons):
    """Tri-state per split: True if every mode confirmed the same global-constant
    asset, False if any mode detected a mismatch, None if the reports predate the
    provenance record (cannot be judged, must not be reported as consistent)."""
    result = {}
    for split, modes in comparisons.items():
        states = {modes[mode]["global_constant_consistent"] for mode in ("clean", "mild") if mode in modes}
        if False in states:
            result[split] = False
        elif states == {True}:
            result[split] = True
        else:
            result[split] = None
    return result


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
    # The global-constant asset must be the same file, from the same manifest,
    # before and after; a mean regenerated mid-run would invalidate the contrast.
    global_before, global_after = before.get("global_constant"), after.get("global_constant")
    if bool(global_before) != bool(global_after):
        raise ValueError("Before/after probe reports disagree on the global-constant provenance record")
    global_consistent = None
    if global_before and global_after:
        if global_before != global_after:
            raise ValueError("Before/after probes used different global-constant assets (file or manifest identity)")
        global_consistent = True
    results = {}
    for mode in ("clean", "mild"):
        a, b = before["aggregate"][mode], after["aggregate"][mode]
        loss = {kind: values["video_loss"] for kind, values in b["variants"].items() if values.get("video_loss") is not None}
        scores = {kind: b["variants"][kind]["relevance_score"] for kind in ("correct", "wrong_source")}
        rows = [c for c in after["cases"] if c["prefix_mode"] == mode]
        target_means = target_level_deltas(rows)
        aggregate_deltas = {name: statistics.mean(t["core_deltas"][name] for t in target_means if t["core_deltas"][name] is not None)
                            if any(t["core_deltas"][name] is not None for t in target_means) else None
                            for name in CORE_DELTAS}
        delta_target_counts = {name: sum(t["core_deltas"][name] is not None for t in target_means) for name in CORE_DELTAS}
        delta_target_positive = {name: sum(t["core_deltas"][name] is not None and t["core_deltas"][name] > 0 for t in target_means)
                                 for name in CORE_DELTAS}
        results[mode] = {"loss_before": {kind: values["video_loss"] for kind, values in a["variants"].items() if values.get("video_loss") is not None},
                         "loss_after": loss, "relevance_before": {k: a["variants"][k]["relevance_score"] for k in scores},
                         "relevance_after": scores,
                         "gate_before": {k: a["variants"][k]["memory_gate_mean"] for k in scores},
                         "gate_after": {k: b["variants"][k]["memory_gate_mean"] for k in scores},
                         "global_constant_before": global_before, "global_constant_after": global_after,
                         "global_constant_consistent": global_consistent,
                         "correct_change_vs_none_percent": (loss["correct"] / loss["none"] - 1) * 100,
                         "wrong_change_vs_none_percent": (loss["wrong_source"] / loss["none"] - 1) * 100,
                         "correct_change_vs_global_constant_percent": ((loss["correct"] / loss["global_constant"] - 1) * 100) if "global_constant" in loss else None,
                         "content_gain_global_constant_minus_correct": (loss["global_constant"] - loss["correct"]) if "global_constant" in loss else None,
                         "constant_change_vs_none_percent": ((loss["global_constant"] / loss["none"] - 1) * 100) if "global_constant" in loss else None,
                         "active_zero_change_vs_none_percent": ((loss["active_zero"] / loss["none"] - 1) * 100) if "active_zero" in loss else None,
                         "pair_mean_change_vs_none_percent": ((loss["pair_mean"] / loss["none"] - 1) * 100) if "pair_mean" in loss else None,
                         "mean_loss_ordered": all(kind in loss for kind in ("correct", "none", "wrong_source")) and loss["correct"] < loss["none"] < loss["wrong_source"],
                         "score_ordered_pairs": b["score_correct_gt_wrong"],
                         "loss_ordered_pairs": b["loss_correct_lt_none_lt_wrong"],
                         "target_noise_pairs": len(rows), "target_means": target_means,
                         "loss_ordered_targets": sum(t["ordered"] for t in target_means),
                         "core_deltas": aggregate_deltas, "core_delta_targets": delta_target_counts,
                         "core_delta_targets_positive": delta_target_positive}
    clean = results["clean"]["loss_after"]["none"]
    results["degradation_effect_on_none_percent"] = (results["mild"]["loss_after"]["none"] / clean - 1) * 100
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "checkpoints/reality_memory_paired_review")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/paired_review")
    args = parser.parse_args()
    args.run, args.output = args.run.resolve(), args.output.resolve()
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
    constant_consistent = asset_consistency(comparisons)
    report = {"batch_step": step, "optimizer_step": updates, "optimizer_state_step_verified": True,
              "sample_visits": dict(Counter(r["sample_id"] for r in rows)), "config": state["config"],
              "comparisons": comparisons, "train_val_sources_disjoint": True,
              "global_constant_asset_consistent": constant_consistent,
              "performance": {"sum_training_step_seconds": sum(r["step_time"] for r in rows),
                              "median_step_seconds": statistics.median(r["step_time"] for r in rows),
                              "peak_vram_bytes": max(r["peak_vram_bytes"] for r in rows)},
              "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
              "scope": "One training initialization, 16 training and 4 held-out targets, two noise seeds per target, first-future-block teacher forcing. "
                       "Core deltas are paired per target+noise and aggregated over unique targets. Supervised relevance ordering alone is not evidence of video utility."}
    write_json(args.output / "summary.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8), constrained_layout=True)
    order = ("U_correct", "G_branch", "G_generic", "G_content")
    colors = ("#347ba5", "#7f9f4f", "#c08b3a", "#8b3a62")
    labels = [f"{split}/{mode}" for split in ("train", "val") for mode in ("clean", "mild")]
    width = .2
    for offset, name, color in zip((-1.5, -.5, .5, 1.5), order, colors):
        values = [comparisons[s][m]["core_deltas"].get(name) for s in ("train", "val") for m in ("clean", "mild")]
        draw = [value if value is not None else 0 for value in values]
        axes[0].bar([i + offset * width for i in range(4)], draw, width, label=name, color=color)
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set_xticks(range(4), labels)
    axes[0].set_ylabel("Video-loss delta (baseline minus variant)")
    axes[0].set_title("Paired target-level core deltas; positive = variant better")
    axes[0].legend(fontsize=8)
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