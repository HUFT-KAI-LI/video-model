"""Build a review report from the completed short run; does not execute any model."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import av
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "checkpoints/reality_memory_overfit_review")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/overfit_review")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in (args.run / "train_rank_0.jsonl").read_text().splitlines()]
    step = records[-1]["step"]
    before_path = args.run / "probe_before_step_0000_rank_0.json"
    after_path = args.run / f"probe_after_step_{step:04d}_rank_0.json"
    before, after = read(before_path), read(after_path)
    if [(c["sample_id"], c["noise_seed"]) for c in before["cases"]] != [(c["sample_id"], c["noise_seed"]) for c in after["cases"]]:
        raise ValueError("Before/after probes do not use the same samples and noise")
    evaluation_dir = ROOT / f"outputs/reality_memory/{args.run.name}/step_{step}/rank_0"
    evaluation = read(evaluation_dir / "metrics.json")
    base = after["aggregate"]["base"]["video_loss"]
    paired = {}
    for kind in ("async", "aligned", "wrong_source"):
        entries = [(case["variants"][kind]["video_loss"], case["variants"]["base"]["video_loss"]) for case in after["cases"]]
        paired[kind] = {"before_video_loss": before["aggregate"][kind]["video_loss"],
                        "after_video_loss": after["aggregate"][kind]["video_loss"],
                        "relative_change_vs_base_percent": (after["aggregate"][kind]["video_loss"] / base - 1) * 100,
                        "wins_vs_base": sum(candidate < reference for candidate, reference in entries),
                        "gate_before": before["aggregate"][kind]["memory_gate_mean"],
                        "gate_after": after["aggregate"][kind]["memory_gate_mean"],
                        "normalized_entropy_after": after["aggregate"][kind]["memory_attention_entropy_normalized"]}
    total_time = sum(r["step_time"] for r in records)
    trace = ROOT / "logs/reality_memory/overfit_review/gpu_utilization.csv"
    with trace.open() as stream:
        utilization = [float(row[2].strip().split()[0]) for row in list(csv.reader(stream))[1:] if len(row) == 4]
    report = {"batch_steps": step, "optimizer_steps": records[-1]["optimizer_steps"],
              "sample_types_seen": dict(Counter(r["reference_kind"] for r in records)),
              "fixed_noise_probe_samples": len(after["cases"]), "paired_controls": paired,
              "no_memory_exact_base": all(c["variants"]["none"]["video_loss"] == c["variants"]["base"]["video_loss"] for c in after["cases"]),
              "correct_vs_wrong_video_loss_gap": after["correct_vs_wrong_video_loss_gap"],
              "ar_validation_cases": len(evaluation["cases"]), "ar_validation": evaluation["aggregate"],
              "performance": {"sum_training_step_seconds": total_time,
                              "median_step_seconds": statistics.median(r["step_time"] for r in records),
                              "data_fraction": sum(r["data_time"] for r in records) / total_time,
                              "peak_vram_bytes": max(r["peak_vram_bytes"] for r in records),
                              "gpu_utilization_process_mean_percent": statistics.mean(utilization),
                              "gpu_utilization_samples": len(utilization),
                              "gpu_scope": "1Hz trace of entire process including model loading, before/after probes, training and AR evaluation"},
              "decision": "Do not expand to 50/200 steps: no correct-reference advantage yet; this short clean-prefix proxy does not reject the long-horizon hypothesis."}
    for path, name in [(before_path, "probe_before.json"), (after_path, "probe_after.json"),
                       (evaluation_dir / "metrics.json", "ar_validation.json"),
                       (args.run / "train_rank_0.jsonl", "train_steps.jsonl")]:
        (args.output / name).write_bytes(path.read_bytes())
    for row in evaluation["cases"]:
        if row["variants"]["no_memory"]["max_abs_difference_from_base"] != 0:
            raise ValueError("No-memory AR invariant failed")
    videos = []
    for path in sorted(evaluation_dir.glob("case_*/*.mp4")):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = sum(1 for _ in container.decode(stream))
            if frames != 57 or float(stream.average_rate) != 16:
                raise ValueError(f"Malformed video: {path}")
        videos.append({"path": str(path.relative_to(ROOT)), "frames": frames,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    report["videos"] = videos
    write_json(args.output / "summary.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ("Async", "Aligned", "Wrong source")
    kinds = ("async", "aligned", "wrong_source")
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].bar(names, [paired[k]["relative_change_vs_base_percent"] for k in kinds], color=["#4178b5", "#489675", "#b57958"])
    axes[0].axhline(0, color="black", linewidth=.7)
    axes[0].set_ylabel("Video loss change vs Base (%)")
    axes[0].set_title("Fixed noise, 16 training targets; lower is better")
    for i, kind in enumerate(kinds):
        axes[1].plot(["Before", "After"], [paired[kind]["gate_before"], paired[kind]["gate_after"]], marker="o", label=names[i])
    axes[1].set_ylabel("Mean relevance gate")
    axes[1].set_title("All three gates decreased")
    axes[1].legend()
    figure.savefig(args.output / "overfit_controls.png", dpi=160)
    plt.close(figure)
    labels = ("base", "hard_anchor", "aligned_k4", "async_k4", "wrong_source_k4")
    frame_indices = (20, 32, 56)
    sheet = Image.new("RGB", (5 * 324, 3 * 210 + 30), "white")
    draw = ImageDraw.Draw(sheet)
    for col, label in enumerate(labels):
        draw.text((col * 324 + 6, 8), label, fill="black")
        with av.open(str(evaluation_dir / "case_000" / (label + ".mp4"))) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in frame_indices:
                    row = frame_indices.index(index)
                    sheet.paste(frame.to_image().resize((324, 192)), (col * 324, row * 210 + 30))
                    draw.text((col * 324 + 5, row * 210 + 224), f"frame {index} / {index / 16:.2f}s", fill="black")
    sheet.save(args.output / "rollout_contact_sheet.jpg", quality=88)
    print(json.dumps({"paired_controls": paired, "performance": report["performance"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
