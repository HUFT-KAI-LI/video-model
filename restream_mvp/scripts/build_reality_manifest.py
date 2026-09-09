"""Build split-safe weak-reference tasks from continuous shots of existing Youku videos."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import read_manifest
from restream.reality_data import canonical_hash, histogram, read_reference, scene_similarity, write_json
from restream.reality_selection import (SELECTION_IDENTITY_SCHEMA, selection_config_hash,
                                        validate_selection_protocol, validate_temporal_sampling)
from restream.runtime import read_config


def reference(row, at, role, shot_id):
    return {"video": row["video"], "video_sha256": row["sha256"], "source_id": row["source_id"],
            "split": row["split"], "time": round(at, 6), "role": role, "shot_id": shot_id}


def selection_times(start, length, arrival, protocol):
    """Times visible to reference filtering under the declared protocol."""
    if protocol == "strict_online":
        return np.linspace(start, start + arrival, 2)
    if protocol == "offline_target_filtered":
        return np.linspace(start, start + length, 3)
    raise ValueError("Unknown reference selection protocol")


def candidate(row, shots, config):
    refs = config["reality_memory"]["references"]
    length = (config["data"]["frames"] - 1) / config["data"]["fps"]
    margin, gap = refs["boundary_margin_sec"], refs["min_gap_sec"]
    rng = random.Random(f"{config['data']['selection_seed']}:{row['source_id']}")
    # Require room for past observations, even when 'both' is enabled.
    eligible = [(i, shot) for i, shot in enumerate(shots)
                if shot["end"] - shot["start"] >= length + 2 * margin + gap + .5]
    rng.shuffle(eligible)
    max_k = refs["pool_size"]
    for shot_id, shot in eligible:
        start = rng.uniform(shot["start"] + margin + gap + .5, shot["end"] - margin - length)
        arrival = 4 * (config["reality_memory"]["objective"]["prefix_latents"] - 1) / config["data"]["fps"]
        protocol = refs.get("selection_protocol", "offline_target_filtered")
        validate_selection_protocol(protocol, refs["async_direction"])
        temporal_sampling = config["data"]["temporal_sampling"]
        validate_temporal_sampling(protocol, temporal_sampling)
        visible_until = None
        if protocol == "strict_online":
            # Every target-prefix analysis frame uses the same causal_previous
            # policy as the Dataset (last frame at/before the requested time), so
            # a sparse or irregular source cannot push an analysis frame past the
            # visible cutoff.
            times = selection_times(start, length, arrival, protocol)
            histograms, sampled = [], []
            for at in times:
                request = reference(row, at, "analysis", shot_id)
                rgb, actual = read_reference(request, 64, return_time=True, latest=True)
                if actual > start + arrival + 1e-6:
                    raise ValueError("strict_online analysis frame exceeds the theoretical visible cutoff")
                histograms.append(histogram(rgb))
                sampled.append(actual)
            visible_until = sampled[-1]
            if visible_until < start - 1e-6:
                continue  # No frame inside the prefix window at this frame rate.
            target_hist = np.mean(histograms, axis=0)
        else:
            target_hist = np.mean([histogram(read_reference(reference(row, at, "analysis", shot_id), 64))
                                   for at in selection_times(start, length, arrival, protocol)], axis=0)
        async_refs = []
        for _ in range(max_k * 8):
            intervals = [(shot["start"] + margin, start - gap)]
            if refs["async_direction"] == "both" and start + length + gap < shot["end"] - margin:
                intervals.append((start + length + gap, shot["end"] - margin))
            low, high = rng.choice(intervals)
            ref = reference(row, rng.uniform(low, high), "same_source_async", shot_id)
            rgb, actual = read_reference(ref, 64, return_time=True)
            ref["time"] = round(actual, 6)
            if not (low <= actual <= high) or any(item["time"] == ref["time"] for item in async_refs):
                continue
            score = scene_similarity(target_hist, histogram(rgb))
            if score >= config["reality_memory"]["filter"]["scene_similarity"]:
                ref["scene_similarity"] = score
                async_refs.append(ref)
            if len(async_refs) == max_k:
                break
        if len(async_refs) != max_k:
            continue
        radius = refs["near_radius_sec"]
        strict = protocol == "strict_online"
        upper = visible_until if strict else min(start + length, start + arrival + radius)
        low = max(start, start + arrival - radius)
        if upper < low:
            continue  # Coarse or missing frames leave no room for a near-aligned reference.
        aligned = []
        for _ in range(max_k * 8):
            at = rng.uniform(low, upper)
            ref = reference(row, at, "near_aligned_soft", shot_id)
            rgb, actual = read_reference(ref, 64, return_time=True, latest=strict)
            ref["time"] = round(actual, 6)
            if actual > upper or any(item["time"] == ref["time"] for item in aligned):
                continue
            score = scene_similarity(target_hist, histogram(rgb))
            if score >= config["reality_memory"]["filter"]["scene_similarity"]:
                ref["scene_similarity"] = score
                aligned.append(ref)
            if len(aligned) == max_k:
                break
        if len(aligned) != max_k:
            continue
        if strict:
            if any(ref["time"] > visible_until + 1e-6 for ref in async_refs + aligned):
                raise ValueError("strict_online selected a reference after visible_until")
        protocol_hash = selection_config_hash(config)
        return {**row, "window_start": start, "window_sec": length, "anchor_sec": [arrival],
                "target_start": start, "target_sec": length, "shot_id": shot_id, "shot": shot,
                "reference_sets": {"async": async_refs, "aligned": aligned},
                "selection_protocol": protocol, "selection_schema": SELECTION_IDENTITY_SCHEMA,
                "temporal_sampling": temporal_sampling,
                "visible_until": round(visible_until, 6) if strict else None,
                "selection_config_hash": protocol_hash,
                "filter_status": "heuristic_pass", "visual_review": "pending",
                "reference_semantics": "same-source continuous-shot proxy; not certified same-world"}
    return None


def attach_references(rows, config):
    """Wrong references never cross the existing train/val source boundary."""
    if len({r["source_id"] for r in rows}) < 2:
        raise ValueError("Need at least two retained sources in each split")
    settings = config["reality_memory"]["references"]
    names = ("async", "aligned", "none", "wrong")
    probabilities = [settings[k] for k in ("async_probability", "aligned_probability", "no_memory_probability", "wrong_memory_probability")]
    if not np.isclose(sum(probabilities), 1) or min(probabilities) < 0:
        raise ValueError("Reference mixture probabilities must sum to one")
    counts = [int(len(rows) * p) for p in probabilities]
    for i in np.argsort([-(len(rows) * p - n) for p, n in zip(probabilities, counts)])[:len(rows) - sum(counts)]:
        counts[i] += 1
    kinds = [name for name, count in zip(names, counts) for _ in range(count)]
    selection_seed = config["data"]["selection_seed"]
    random.Random(selection_seed).shuffle(kinds)
    for row, kind in zip(rows, kinds):
        rng = random.Random(f"{selection_seed}:refs:{row['source_id']}")
        donors = [item for item in rows if item["source_id"] != row["source_id"]]
        donor = rng.choice(donors)
        row["reference_sets"]["wrong"] = [{**ref, "role": "wrong_source"} for ref in donor["reference_sets"]["async"]]
        count = rng.randint(settings["min_count"], settings["max_count"])
        row["reference_kind"] = kind
        row["references"] = [] if kind == "none" else row["reference_sets"][kind][:count]
        row["sample_id"] = canonical_hash({"source": row["source_id"], "start": row["target_start"], "kind": kind})[:24]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--shots", type=Path, default=ROOT / "data/reality_shots.jsonl")
    args = parser.parse_args()
    config = read_config(args.config)
    references = config["reality_memory"]["references"]
    protocol = references.get("selection_protocol", "offline_target_filtered")
    validate_selection_protocol(protocol, references.get("async_direction"))
    validate_temporal_sampling(protocol, config["data"].get("temporal_sampling"))
    if type(config["data"].get("selection_seed")) is not int or config["data"]["selection_seed"] < 0:
        raise ValueError("data.selection_seed must be a nonnegative integer, separate from the training seed")
    pool_size = references.get("pool_size")
    counts = config["eval"]["reference_counts"]
    if type(pool_size) is not int or pool_size < references["max_count"] or pool_size < max(int(count) for count in counts):
        raise ValueError("references.pool_size must be explicit and cover max_count and the evaluation sweep")
    if references["async_direction"] not in ("past_only", "both") or references["min_count"] < 1 or references["max_count"] < references["min_count"]:
        raise ValueError("Invalid reference sampling configuration")
    shots = {r["source_id"]: r for r in read_manifest(args.shots)}
    stats = {"config": config, "splits": {}, "notes": [
        "Ratios apply to retained rows; per-reference dropout is applied during training.",
        "Same-shot and histogram checks are heuristics; scene identity and captions still need human review.",
        "Reference timestamps are metadata only; async defaults to past-only outside a 1.5s gap."]}
    sources = {}
    for split in ("train", "val"):
        rows = read_manifest(ROOT / f"data/{split}.jsonl")
        sources[split] = {r["source_id"] for r in rows}
        def make(row):
            record = shots[row["source_id"]]
            if record["error"]:
                return None
            if record["video_sha256"] != row["sha256"] or record["filter_hash"] != canonical_hash(config["reality_memory"]["filter"]):
                raise ValueError("Stale shot metadata; rerun filter_continuous_shots.py")
            return candidate(row, record["shots"], config)
        retained = []
        with ThreadPoolExecutor(max_workers=config["reality_memory"]["filter"]["workers"]) as pool:
            for i, item in enumerate(pool.map(make, rows)):
                if item is not None:
                    retained.append(item)
                if (i + 1) % 50 == 0:
                    print(f"{split}: examined {i + 1}/{len(rows)}, retained {len(retained)}", flush=True)
        attach_references(retained, config)
        path = ROOT / config["data"][f"{split}_manifest"]
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retained))
        temporary.replace(path)
        stats["splits"][split] = {"input_sources": len(rows), "retained_sources": len(retained),
                                  "rejected_sources": len(rows) - len(retained),
                                  "reference_kinds": dict(Counter(r["reference_kind"] for r in retained))}
    if sources["train"] & sources["val"]:
        raise ValueError("Existing split leaks source IDs")
    write_json(ROOT / "data/reality_stats.json", stats)
    print(json.dumps(stats["splits"], indent=2))


if __name__ == "__main__":
    main()
