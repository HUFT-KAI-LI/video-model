import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
import av
import numpy as np
import torch
from restream.objective import future_loss
from restream.reality_memory import RealityMemory
from restream.dataset import VideoDataset
from restream.reality_dataset import RealityDataset
from restream.reality_paired import (paired_loss, paired_history, paired_references,
                                    contrast_loss, validate_paired_config, load_global_constant,
                                    global_constant_provenance)
from restream.reality_paired_diagnostics import aggregate_pairs
from restream.reality_runtime import PreserveHistory, read_reality_config
from restream.training_budget import TrainingBudget
from restream.reality_selection import (GLOBAL_MEAN_SCHEMA, SELECTION_IDENTITY_SCHEMA,
                                        select_targets, selection_config_hash, selection_identity,
                                        unique_train_reference_pool, validate_temporal_sampling)
from restream.reality_stats import auroc, bootstrap_ci
from restream.reality_data import canonical_hash
from train_reality_memory import paired_diagnostics_enabled

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def minimal_paired_config(train_manifest=None, protocol="offline_target_filtered", global_path=None,
                          pool_size=2, selection_seed=7, temporal_sampling=None):
    """Complete config for selection identity and global-mean provenance checks."""
    return {
        "seed": 7,
        "data": {"frames": 57, "fps": 16.0,
                 "train_manifest": str(train_manifest or (ROOT / "data/reality_train.jsonl")),
                 "selection_seed": selection_seed,
                 "temporal_sampling": temporal_sampling or ("causal_previous" if protocol == "strict_online" else "first_at_or_after")},
        "eval": {"reference_counts": [1, 2]},
        "reality_memory": {
            "references": {"selection_protocol": protocol, "async_direction": "past_only",
                           "min_gap_sec": 1.5, "boundary_margin_sec": 0.2,
                           "near_radius_sec": 0.5, "max_count": 2, "min_count": 1,
                           "pool_size": pool_size,
                           **({"global_constant_features": str(global_path)} if global_path else {})},
            "filter": {"scene_similarity": 0.65, "histogram_cut": 0.5, "pixel_jump": 0.35,
                       "black_level": 16, "black_fraction": 0.95},
            "objective": {"prefix_latents": 6},
        },
    }


def manifest_row(row, video="/tmp/fake.mp4", caption="c"):
    return {"video": video, "sha256": "x", "caption": caption, "source_id": "a", "split": "train",
            "window_start": 0.0, "window_sec": 3.5, "anchor_sec": [1.25], "target_start": 0.0,
            "target_sec": 3.5, "shot_id": 0, "shot": {"start": 0, "end": 5}, "reference_kind": "async",
            "references": [], "sample_id": "s", "filter_status": "pass", "visual_review": "pending",
            **row}


def low_fps_video(folder, seconds=40, rate=4, size=64):
    """Tiny constant-colour video whose frame spacing (1/rate) exceeds the target
    sampling step (1/16), so causal_previous must repeat frames."""
    path = Path(folder) / f"lowfps_{rate}.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=rate)
        stream.width, stream.height, stream.pix_fmt = size, size, "yuv420p"
        for _ in range(seconds * rate):
            pixels = np.full((size, size, 3), 128, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


class PairedTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.config = read_reality_config(ROOT / "configs/reality_memory_paired.yaml")
        self.cfg = self.config["reality_memory"]["objective"]["paired"]

    def test_prefix_is_shared_mild_and_gt_remains_immutable(self):
        gt = torch.randn(1, 15, 4, 16, 16)
        original = gt.clone()
        one = paired_history(gt, 5, self.cfg, "cpu", 13)
        two = paired_history(gt, 5, self.cfg, "cpu", 13)
        self.assertTrue(torch.equal(one, two))
        self.assertTrue(torch.equal(gt, original))
        self.assertTrue(torch.equal(one[:, :3], gt[:, :3]))
        self.assertTrue(torch.equal(paired_history(gt, 5, self.cfg, "cpu", 13, "clean"), gt[:, :6]))
        self.assertAlmostEqual((one[:, 3:] - gt[:, 3:6]).square().mean().item(), .05**2, delta=.0002)

    def test_paired_pools_enforce_count_split_and_source(self):
        ref = {"source_id": "target", "split": "train", "time": 1.}
        row = {"source_id": "target", "split": "train", "target_start": 5.,
               "reference_sets": {"async": [ref.copy(), ref.copy()],
                                  "wrong": [{**ref, "source_id": "donor"}] * 2}}
        correct, wrong = paired_references(row, self.cfg)
        self.assertEqual(len(correct), len(wrong))
        for key, value in (("split", "val"), ("source_id", "target")):
            bad = copy.deepcopy(row)
            bad["reference_sets"]["wrong"][0][key] = value
            with self.assertRaises(ValueError):
                paired_references(bad, self.cfg)
        row["reference_sets"]["async"][0]["time"] = 5.
        with self.assertRaises(ValueError):
            paired_references(row, self.cfg)

    def test_score_supervision_does_not_directly_force_final_gate(self):
        model = RealityMemory(6, 8, 12, 2)
        _, stats = model(torch.randn(1, 5, 12).expand(2, -1, -1), torch.randn(2, 2, 3, 6),
                         torch.ones(2, 2, dtype=torch.bool))
        contrast_loss(*stats["relevance_score"], .1).backward()
        self.assertTrue(all(p.grad is None for p in model.gate.parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.query.parameters()), 0)

    def test_constant_variant_is_reported_separately_from_none_and_wrong(self):
        values = {kind: {"video_loss": value, "relevance_score": 0., "memory_gate_mean": .1}
                  for kind, value in (("base", 1.), ("none", 1.), ("pair_mean", .9), ("global_constant", .9), ("correct", .8), ("wrong_source", .85))}
        cases = [{"sample_id": "x", "prefix_mode": mode, "variants": values} for mode in ("clean", "mild")]
        aggregate = aggregate_pairs(cases)
        self.assertAlmostEqual(aggregate["clean"]["correct_minus_global_constant"], -.1)
        self.assertAlmostEqual(aggregate["clean"]["global_constant_minus_none"], -.1)

    def test_target_level_core_deltas_are_unique_target_means(self):
        # U = L_none - L_correct; G_branch = L_none - L_active_zero;
        # G_generic = L_active_zero - L_global; G_content = L_global - L_correct;
        # S_reference = L_wrong - L_correct; all averaged per unique target.
        def variants_for(value_scale):
            return {"base": {"video_loss": 1.0}, "none": {"video_loss": 1.0},
                    "active_zero": {"video_loss": .9 * value_scale, "relevance_score": 0.3, "memory_gate_mean": .2},
                    "pair_mean": {"video_loss": .85 * value_scale, "relevance_score": 0.3, "memory_gate_mean": .2},
                    "global_constant": {"video_loss": .8 * value_scale, "relevance_score": 0.3, "memory_gate_mean": .2},
                    "global_async": {"video_loss": .82 * value_scale, "relevance_score": 0.3, "memory_gate_mean": .2},
                    "global_aligned": {"video_loss": .84 * value_scale, "relevance_score": 0.3, "memory_gate_mean": .2},
                    "correct": {"video_loss": .6 * value_scale, "relevance_score": 0.5, "memory_gate_mean": .2},
                    "wrong_source": {"video_loss": .95 * value_scale, "relevance_score": 0.4, "memory_gate_mean": .2}}
        cases = []
        for sample_id in ("a", "b"):
            for mode in ("clean", "mild"):
                for seed in (1, 2):
                    cases.append({"sample_id": sample_id, "prefix_mode": mode, "noise_seed": seed,
                                  "variants": variants_for(1.0)})
        aggregate = aggregate_pairs(cases)["clean"]["core_deltas"]
        for name, expected in (("U_correct", .4), ("G_branch", .1), ("G_generic", .1),
                               ("G_content", .2), ("G_content_matched", .22), ("S_reference", .35)):
            self.assertAlmostEqual(aggregate[name], expected, places=6)
        self.assertEqual(aggregate_pairs(cases)["clean"]["core_delta_targets"]["U_correct"], 2)

    def test_selection_identity_separates_seed_sampling_and_shot_filter(self):
        base = minimal_paired_config()
        reference = selection_config_hash(base)
        for path, value in ((("data", "selection_seed"), 99),
                            (("reality_memory", "references", "pool_size"), 4),
                            (("reality_memory", "filter", "histogram_cut"), 0.4)):
            changed = copy.deepcopy(base)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            self.assertNotEqual(selection_config_hash(changed), reference, msg=str(path))
        # Training-seed, mixture probabilities and compute knobs must not change it.
        for path, value in ((("seed",), 1234),
                            (("reality_memory", "references", "async_probability"), 0.9),
                            (("reality_memory", "filter", "workers"), 64),
                            (("data", "workers"), 8)):
            changed = copy.deepcopy(base)
            target = changed
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            self.assertEqual(selection_config_hash(changed), reference, msg=str(path))
        # The sampling policy is protocol-bound but still part of the identity.
        strict = minimal_paired_config(protocol="strict_online")
        self.assertEqual(selection_identity(strict)["temporal_sampling"], "causal_previous")
        self.assertNotEqual(selection_config_hash(strict), reference)
        self.assertEqual(selection_identity(base)["shot_filter_hash"],
                         canonical_hash({"histogram_cut": 0.5, "pixel_jump": 0.35, "black_level": 16, "black_fraction": 0.95}))

    def test_temporal_sampling_policy_is_fixed_by_protocol(self):
        validate_temporal_sampling("offline_target_filtered", "first_at_or_after")
        validate_temporal_sampling("strict_online", "causal_previous")
        with self.assertRaises(ValueError):
            validate_temporal_sampling("offline_target_filtered", "causal_previous")
        with self.assertRaises(ValueError):
            validate_temporal_sampling("strict_online", "first_at_or_after")
        with self.assertRaises(ValueError):
            validate_temporal_sampling("strict_online", "bad")

    def test_train_positive_pool_validation_rejects_leaks(self):
        identity = {"encoder": "fixture"}
        cache = SimpleNamespace(key=lambda ref: canonical_hash({"video_sha256": ref["video_sha256"], "time": ref["time"], "encoder": identity}))
        good = {"split": "train", "sample_id": "s", "source_id": "a", "shot_id": 0,
                "reference_sets": {"async": [{"split": "train", "source_id": "a", "shot_id": 0, "video_sha256": "v", "time": 1.0}], "aligned": []}}
        self.assertEqual(len(unique_train_reference_pool([good], cache)), 1)
        for path, value in (("split", "val"), ("source_id", "b"), ("shot_id", 1)):
            bad = copy.deepcopy(good)
            bad["reference_sets"]["async"][0][path] = value
            with self.assertRaises(ValueError):
                unique_train_reference_pool([bad], cache)
        with self.assertRaises(ValueError):
            unique_train_reference_pool([{**good, "split": "val"}], cache)

    def test_global_constant_is_fixed_and_provenance_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder / "train.jsonl"
            identity = {"encoder": "fixture", "version": 1}
            shared = {"video_sha256": "v1", "split": "train", "source_id": "a", "shot_id": 0}
            rows = [{"source_id": "a", "split": "train", "shot_id": 0, "sample_id": "s1",
                     "reference_sets": {"async": [{**shared, "time": 1.0}], "aligned": []}},
                    {"source_id": "a", "split": "train", "shot_id": 0, "sample_id": "s2",
                     "reference_sets": {"async": [{**shared, "time": 1.0}, {**shared, "time": 2.0}], "aligned": []}}]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cache = SimpleNamespace(identity=identity, tokens=3, channels=4,
                                    key=lambda ref: canonical_hash({"video_sha256": ref["video_sha256"],
                                                                    "time": ref["time"], "encoder": identity}),
                                    read=lambda ref: torch.arange(12, dtype=torch.float32).reshape(3, 4) * (ref["time"] / 3 + 1))
            path = folder / "mean.pt"
            config = minimal_paired_config(train_manifest=manifest, global_path=path)
            positive = torch.arange(12, dtype=torch.float32).reshape(3, 4) * 1.5
            async_mean = torch.arange(12, dtype=torch.float32).reshape(3, 4) * 1.5
            aligned_mean = torch.full((3, 4), -1.0)
            async_keys = sorted({cache.key(r) for row in rows for r in row["reference_sets"]["async"]})
            torch.save({"schema": GLOBAL_MEAN_SCHEMA, "features": positive,
                        "means": {"async": async_mean, "aligned": aligned_mean, "positive": positive},
                        "roles": {"async": {"unique_reference_count": 2, "reference_keys_sha256": canonical_hash(async_keys)},
                                  "aligned": {"unique_reference_count": 0, "reference_keys_sha256": canonical_hash([])},
                                  "positive": {"unique_reference_count": 2, "reference_keys_sha256": canonical_hash(async_keys)}},
                        "default_role": "positive",
                        "source_split": "train",
                        "cache_identity": identity, "tokens": 3, "channels": 4,
                        "selection_protocol": "offline_target_filtered",
                        "temporal_sampling": config["data"]["temporal_sampling"],
                        "selection_seed": config["data"]["selection_seed"],
                        "train_manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
                        "selection_config_hash": selection_config_hash(config)}, path)
            first = load_global_constant(config, cache, "cpu")
            second = load_global_constant(config, cache, "cpu", "async")
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(first, positive))
            self.assertTrue(torch.equal(load_global_constant(config, cache, "cpu", "aligned"), aligned_mean))
            record = global_constant_provenance(config, cache, "cpu")
            self.assertEqual(record["roles"]["positive"]["unique_reference_count"], 2)
            self.assertEqual(record["roles"]["async"]["unique_reference_count"], 2)
            with self.assertRaises(ValueError):
                load_global_constant(config, cache, "cpu", "missing_role")
            cache.identity = {"encoder": "other"}
            with self.assertRaises(ValueError):
                load_global_constant(config, cache, "cpu")
            cache.identity = identity
            # Regenerating the train manifest must invalidate the stored control.
            manifest.write_text(json.dumps({"source_id": "a", "split": "train", "shot_id": 0, "sample_id": "s1",
                                            "reference_sets": {"async": [{**shared, "time": 9.0}], "aligned": []}}) + "\n")
            with self.assertRaises(ValueError):
                load_global_constant(config, cache, "cpu")

    def test_memory_reports_raw_and_applied_delta_norm(self):
        model = RealityMemory(6, 8, 12, 2)
        context = torch.randn(1, 5, 12)
        fused, stats = model(context, torch.randn(1, 2, 3, 6), torch.ones(1, 2, dtype=torch.bool))
        self.assertIn("raw_delta_norm", stats)
        self.assertIn("applied_delta_norm", stats)
        torch.testing.assert_close(stats["applied_delta_norm"], torch.zeros_like(stats["applied_delta_norm"]))

    def test_strict_online_selection_never_reads_target_future(self):
        module = script("build_reality_manifest")
        strict = module.selection_times(10., 3.5, .75, "strict_online")
        offline = module.selection_times(10., 3.5, .75, "offline_target_filtered")
        self.assertLessEqual(max(strict), 10.75)
        self.assertGreater(max(offline), 10.75)
        with self.assertRaises(ValueError):
            module.selection_times(10., 3.5, .75, "bad")
        with self.assertRaises(ValueError):
            module.validate_selection_protocol("strict_online", "both")
        module.validate_selection_protocol("strict_online", "past_only")

    def test_candidate_strict_online_full_path_is_causal_and_hash_bound(self):
        """Run the real candidate() under a recording decoder and verify that every
        analysis frame and every returned positive reference stays inside the
        visible prefix, that all analysis reads use causal_previous, and that the
        row carries the shared selection hash."""
        module = script("build_reality_manifest")
        step = 1 / 16.0
        calls = []

        def recording_decoder(reference, size=None, return_time=False, latest=False):
            index = math.floor(reference["time"] / step + 1e-9) if latest else math.ceil(reference["time"] / step - 1e-9)
            actual = float(index * step)
            calls.append({"time": reference["time"], "actual": actual, "latest": latest, "role": reference.get("role")})
            rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
            return (rgb, actual) if return_time else rgb

        module.read_reference = recording_decoder
        row = {"source_id": "s1", "video": "/tmp/fake.mp4", "sha256": "deadbeef", "split": "train"}
        shots = [{"start": 0.0, "end": 40.0}]
        prefix_latents, fps = 6, 16.0
        config = minimal_paired_config(protocol="strict_online", pool_size=4)
        config["reality_memory"]["references"]["max_count"] = 4
        config["eval"]["reference_counts"] = [1, 2, 4]
        item = module.candidate(row, shots, config)
        self.assertIsNotNone(item)
        nominal = item["target_start"] + 4 * (prefix_latents - 1) / fps
        visible = item["visible_until"]
        self.assertIsNotNone(visible)
        self.assertLessEqual(visible, nominal + 1e-6)
        self.assertGreaterEqual(visible, item["target_start"] - 1e-6)
        analysis = [c for c in calls if c["role"] == "analysis"]
        self.assertTrue(analysis)
        self.assertTrue(all(c["latest"] for c in analysis), "Every strict analysis frame must be read at-or-before")
        self.assertTrue(all(c["actual"] <= nominal + 1e-6 for c in analysis))
        for kind in ("async", "aligned"):
            self.assertTrue(item["reference_sets"][kind], f"Expected {kind} references")
            for ref in item["reference_sets"][kind]:
                self.assertLessEqual(ref["time"], visible + 1e-6)
        self.assertEqual(item["selection_protocol"], "strict_online")
        self.assertEqual(item["temporal_sampling"], "causal_previous")
        self.assertEqual(item["selection_config_hash"], selection_config_hash(config))
        self.assertEqual(item["selection_schema"], SELECTION_IDENTITY_SCHEMA)

    def test_candidate_offline_still_filters_with_full_target_histogram(self):
        module = script("build_reality_manifest")
        calls = []

        def recording_decoder(reference, size=None, return_time=False, latest=False):
            calls.append({"time": reference["time"], "latest": latest, "role": reference.get("role")})
            rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
            return (rgb, float(reference["time"])) if return_time else rgb

        module.read_reference = recording_decoder
        row = {"source_id": "s1", "video": "/tmp/fake.mp4", "sha256": "deadbeef", "split": "train"}
        shots = [{"start": 0.0, "end": 40.0}]
        config = minimal_paired_config(protocol="offline_target_filtered", pool_size=4)
        config["reality_memory"]["references"]["max_count"] = 4
        config["eval"]["reference_counts"] = [1, 2, 4]
        item = module.candidate(row, shots, config)
        self.assertIsNotNone(item)
        self.assertIsNone(item["visible_until"])
        self.assertEqual(item["temporal_sampling"], "first_at_or_after")
        arrival = 4 * (config["reality_memory"]["objective"]["prefix_latents"] - 1) / config["data"]["fps"]
        analysis = [c for c in calls if c["role"] == "analysis"]
        self.assertGreater(max(c["time"] for c in analysis), item["target_start"] + arrival,
                           "offline filtering legitimately reads the full target histogram")
        self.assertFalse(any(c["latest"] for c in analysis))
        self.assertEqual(item["selection_config_hash"], selection_config_hash(config))

    def test_low_framerate_strict_dataset_matches_manifest_visible_until(self):
        """The Dataset's causal_previous prefix must consume exactly the frame the
        manifest declared as visible_until, even on a low-fps source."""
        module = script("build_reality_manifest")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = low_fps_video(folder)
            row = {"source_id": "s1", "video": str(video), "sha256": "x", "split": "train"}
            config = minimal_paired_config(protocol="strict_online", pool_size=1)
            config["reality_memory"]["references"]["max_count"] = 1
            config["eval"]["reference_counts"] = [1]
            item = module.candidate(row, [{"start": 0.0, "end": 40.0}], config)
            self.assertIsNotNone(item)
            manifest = folder / "m.jsonl"
            manifest.write_text(json.dumps(manifest_row(item)) + "\n")
            dataset = VideoDataset(manifest, 57, 256, 432, 16, temporal_sampling="causal_previous")
            times = dataset[0]["sampled_times"]
            boundary = 4 * (config["reality_memory"]["objective"]["prefix_latents"] - 1)
            self.assertAlmostEqual(times[boundary].item(), item["visible_until"], places=5)
            # Frames after the boundary are the supervised future, not the prefix.
            self.assertLessEqual(times[:boundary + 1].max().item(), item["visible_until"] + 1e-6)
            self.assertGreaterEqual(times.min().item(), 0.0)

    def test_strict_runtime_prefix_assertion_rejects_tampered_visible_until(self):
        """RealityDataset must verify at decode time that the prefix end equals the
        manifest visible_until, not merely that the manifest is internally bounded."""
        module = script("build_reality_manifest")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video = low_fps_video(folder)
            row = {"source_id": "s1", "video": str(video), "sha256": "x", "split": "train"}
            config = minimal_paired_config(protocol="strict_online", pool_size=1)
            config["reality_memory"]["references"]["max_count"] = 1
            config["eval"]["reference_counts"] = [1]
            item = module.candidate(row, [{"start": 0.0, "end": 40.0}], config)
            self.assertIsNotNone(item)
            cache = SimpleNamespace(identity="x", tokens=1, channels=1, path=lambda ref: Path("/nonexistent"))
            manifest = folder / "m.jsonl"
            good = manifest_row(item, video=str(video))
            manifest.write_text(json.dumps(good) + "\n")

            def load():
                return RealityDataset(manifest, cache, preflight=False, selection_protocol="strict_online",
                                      selection_config_hash=item["selection_config_hash"], prefix_latents=6,
                                      temporal_sampling="causal_previous")

            load()[0]  # Consistent manifest and decode.
            arrival = 4 * (config["reality_memory"]["objective"]["prefix_latents"] - 1) / config["data"]["fps"]
            tampered = min(good["visible_until"] + 0.01, item["target_start"] + arrival - 1e-3)
            self.assertNotAlmostEqual(tampered, good["visible_until"], places=6)
            manifest.write_text(json.dumps({**good, "visible_until": round(tampered, 6)}) + "\n")
            with self.assertRaises(ValueError):
                load()[0]

    def test_global_mean_pool_is_deduplicated_and_mixture_independent(self):
        module = script("cache_reality_features")
        identity = {"encoder": "fixture", "version": 1}
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "train.jsonl"
            ref = lambda time: {"video_sha256": "v", "time": time, "split": "train", "source_id": "a", "shot_id": 0}
            rows = [{"source_id": "a", "split": "train", "shot_id": 0, "sample_id": "s1",
                     "reference_sets": {"async": [ref(1.0), ref(2.0)], "aligned": []}},
                    {"source_id": "a", "split": "train", "shot_id": 0, "sample_id": "s2",
                     "reference_sets": {"async": [ref(1.0)], "aligned": [ref(2.0)]}}]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cache = SimpleNamespace(identity=identity, tokens=3, channels=4,
                                    key=lambda reference: canonical_hash({"video_sha256": reference["video_sha256"],
                                                                          "time": reference["time"], "encoder": identity}),
                                    read=lambda reference: torch.arange(12, dtype=torch.float32).reshape(3, 4) * reference["time"])
            config = minimal_paired_config(train_manifest=manifest)
            payload = module.global_mean_payload(rows, cache, config)
            self.assertEqual(payload["roles"]["positive"]["unique_reference_count"], 2)
            self.assertEqual(payload["roles"]["async"]["unique_reference_count"], 2)
            self.assertEqual(payload["roles"]["aligned"]["unique_reference_count"], 1)
            torch.testing.assert_close(payload["means"]["positive"], torch.arange(12, dtype=torch.float32).reshape(3, 4) * 1.5)
            torch.testing.assert_close(payload["means"]["async"], torch.arange(12, dtype=torch.float32).reshape(3, 4) * 1.5)
            self.assertEqual(payload["schema"], GLOBAL_MEAN_SCHEMA)
            self.assertEqual(payload["source_split"], "train")
            self.assertEqual(payload["temporal_sampling"], config["data"]["temporal_sampling"])
            self.assertEqual(payload["train_manifest_sha256"], __import__("hashlib").sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(payload["selection_config_hash"], selection_config_hash(config))
            with self.assertRaises(ValueError):
                module.global_mean_payload([{**rows[0], "split": "val"}], cache, config)

    def test_reality_dataset_enforces_strict_online_bounds_and_selection_hash(self):
        def refs(*times):
            return {"async": [{"split": "train", "source_id": "a", "shot_id": 0, "time": t} for t in times], "aligned": []}

        cache = SimpleNamespace(identity="x", tokens=1, channels=1, path=lambda ref: Path("/nonexistent"))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "m.jsonl"

            def load(rows, protocol=None, selection_hash=None, temporal=None, legacy=False):
                path.write_text("".join(json.dumps(manifest_row(row)) + "\n" for row in rows))
                return RealityDataset(path, cache, preflight=False, selection_protocol=protocol,
                                      selection_config_hash=selection_hash, prefix_latents=6,
                                      temporal_sampling=temporal, allow_legacy_offline_manifest=legacy)

            good = {"selection_protocol": "strict_online", "selection_schema": SELECTION_IDENTITY_SCHEMA,
                    "temporal_sampling": "causal_previous", "visible_until": 1.0,
                    "reference_sets": refs(.5, .9), "selection_config_hash": "expected-hash"}
            load([good], protocol="strict_online", selection_hash="expected-hash", temporal="causal_previous")
            for broken in ({**good, "selection_schema": 999},
                           {**good, "selection_config_hash": "stale"},
                           {**good, "selection_config_hash": None},
                           {**good, "visible_until": None},
                           {**good, "visible_until": 2.0},
                           {**good, "temporal_sampling": "first_at_or_after"},
                           {**good, "reference_sets": refs(.5, 1.5)},
                           {**good, "reference_sets": {"async": [{"split": "val", "source_id": "a", "shot_id": 0, "time": .5}], "aligned": []}},
                           {**good, "reference_sets": {"async": [{"split": "train", "source_id": "b", "shot_id": 0, "time": .5}], "aligned": []}},
                           {**good, "reference_sets": {"async": [{"split": "train", "source_id": "a", "shot_id": 3, "time": .5}], "aligned": []}}):
                with self.assertRaises(ValueError, msg=str(broken.get("visible_until"))):
                    load([broken], protocol="strict_online", selection_hash="expected-hash", temporal="causal_previous")
            with self.assertRaises(ValueError):
                load([good], protocol="strict_online", selection_hash=None, temporal="causal_previous")
            # Legacy offline rows need the explicit opt-in; new-format offline rows are hashed.
            legacy_row = {"reference_sets": refs(.5)}
            with self.assertRaises(ValueError):
                load([legacy_row], protocol="offline_target_filtered")
            load([legacy_row], protocol="offline_target_filtered", legacy=True)
            new_offline = {"selection_protocol": "offline_target_filtered", "selection_schema": SELECTION_IDENTITY_SCHEMA,
                           "temporal_sampling": "first_at_or_after", "visible_until": None,
                           "reference_sets": refs(.5), "selection_config_hash": "expected-hash"}
            load([new_offline], protocol="offline_target_filtered", selection_hash="expected-hash", temporal="first_at_or_after")
            with self.assertRaises(ValueError):
                load([{**new_offline, "selection_config_hash": "stale"}], protocol="offline_target_filtered",
                     selection_hash="expected-hash", temporal="first_at_or_after")

    def test_paired_gradient_diagnostics_schedule_is_off_by_one_free(self):
        cfg = {"diagnostic_gradients": True, "diagnostic_interval": 5}
        enabled = [step for step in range(1, 12) if paired_diagnostics_enabled(cfg, step)]
        self.assertEqual(enabled, [1, 2, 5, 10])
        self.assertFalse(paired_diagnostics_enabled({"diagnostic_gradients": False, "diagnostic_interval": 5}, 1))
        self.assertFalse(paired_diagnostics_enabled({"diagnostic_gradients": False, "diagnostic_interval": 5}, 5))

    def test_global_constant_asset_consistency_is_tri_state(self):
        module = script("summarize_reality_paired")
        consistent = {"train": {"clean": {"global_constant_consistent": True}, "mild": {"global_constant_consistent": True}}}
        legacy = {"train": {"clean": {"global_constant_consistent": None}, "mild": {"global_constant_consistent": None}}}
        mixed = {"train": {"clean": {"global_constant_consistent": True}, "mild": {"global_constant_consistent": False}}}
        self.assertEqual(module.asset_consistency(consistent), {"train": True})
        self.assertEqual(module.asset_consistency(legacy), {"train": None})
        self.assertEqual(module.asset_consistency(mixed), {"train": False})

    def test_seeded_target_selection_is_deterministic(self):
        rows = [{"sample_id": str(index)} for index in range(10)]
        self.assertEqual(select_targets(rows, 4, 5), select_targets(rows, 4, 5))
        self.assertEqual(len(select_targets(rows, 4, 5)), 4)
        self.assertNotEqual(select_targets(rows, 4, 5), select_targets(rows, 4, 6))
        self.assertEqual(select_targets(rows, 0, 5), list(range(10)))
        self.assertEqual(select_targets(rows, 99, 5), list(range(10)))

    def test_prefix_retrieval_helpers(self):
        module = script("check_prefix_reference_retrieval")
        self.assertEqual(module.prefix_frame_indices(24, 3), [0, 12, 23])
        self.assertEqual(module.prefix_frame_indices(24, 1), [23])
        with self.assertRaises(ValueError):
            module.prefix_frame_indices(24, 0)
        recall = module.gallery_recall([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0], (1, 2))
        self.assertAlmostEqual(recall["recall@1"], .5)
        self.assertAlmostEqual(recall["recall@2"], 1.0)
        self.assertTrue(recall["top1_correct"])
        stats = None
        self.assertAlmostEqual(auroc([3, 2, 1, 0], [1, 1, 0, 0]), 1.0)
        self.assertAlmostEqual(auroc([0, 1, 2, 3], [1, 1, 0, 0]), 0.0)
        self.assertAlmostEqual(auroc([1, 1, 1, 1], [1, 1, 0, 0]), 0.5)
        self.assertIsNone(auroc([1, 2], [1, 1]))
        ci = bootstrap_ci([1.0, 2.0, 3.0], samples=200, seed=0)
        self.assertEqual(ci["n"], 3)
        self.assertLessEqual(ci["low"], ci["mean"])
        self.assertGreaterEqual(ci["high"], ci["mean"])

    def test_hard_negative_is_cross_source_and_feature_selected(self):
        module = script("check_prefix_reference_retrieval")
        rows = [{"sample_id": "t", "source_id": "A"}, {"sample_id": "near", "source_id": "B"},
                {"sample_id": "far", "source_id": "C"}, {"sample_id": "same", "source_id": "A"}]
        row_means = torch.tensor([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [1.0, 0.0]])
        chosen = module.choose_hard_negative(0, rows, row_means, torch.tensor([1.0, 0.0]), "cpu")
        self.assertEqual(chosen, 1)  # Most similar cross-source row, never the same source.

    def test_existing_probe_summary_uses_unique_targets_and_bootstrap(self):
        module = script("summarize_existing_probe")
        def case(sample_id, mode, seed, losses):
            variants = {kind: {"video_loss": value} for kind, value in losses.items()}
            variants["base"] = {"video_loss": losses["none"]}
            return {"sample_id": sample_id, "source_id": sample_id, "prefix_mode": mode, "noise_seed": seed,
                    "variants": variants}
        losses = {"none": 1.0, "active_zero": .95, "pair_mean": .9, "global_constant": .85,
                  "global_async": .83, "global_aligned": .87, "correct": .7, "wrong_source": 1.05}
        report = {"split": "val", "config": {"reality_memory": {"objective": {"paired": {"correct_kind": "async"}}}},
                  "cases": [case("t1", mode, seed, losses) for mode in ("clean", "mild") for seed in (1, 2)]}
        summary = module.summarize(report, samples=200, seed=3)
        self.assertEqual(summary["modes"]["clean"]["unique_targets"], 1)
        self.assertEqual(summary["correct_kind"], "async")
        deltas = summary["modes"]["clean"]["core_deltas"]
        self.assertAlmostEqual(deltas["U_correct"]["mean"], .3, places=6)
        self.assertAlmostEqual(deltas["G_content"]["mean"], .15, places=6)
        self.assertAlmostEqual(deltas["G_content_matched"]["mean"], .13, places=6)
        self.assertEqual(deltas["U_correct"]["n"], 1)
        self.assertLessEqual(deltas["U_correct"]["low"], deltas["U_correct"]["mean"])
        self.assertGreaterEqual(deltas["U_correct"]["high"], deltas["U_correct"]["mean"])

    def test_probe_checkpoint_verification_handles_stored_extra_fields(self):
        """Regression: the stored signature carries selected_indices/world_size, so
        only the semantic fields may be compared; evaluation-only config changes
        are allowed, semantic ones are not."""
        probe = script("probe_existing_memory_checkpoint")
        from restream.reality_runtime import resume_signature
        config = read_reality_config(ROOT / "configs/reality_memory_paired.yaml")
        cache = SimpleNamespace(identity={"encoder": "fixture", "weights": {}})
        saved = {"stage": "r0", "config": copy.deepcopy(config),
                 "signature": {**resume_signature(config, cache), "selected_indices": [1, 2], "world_size": 1}}
        self.assertTrue(probe.verify_checkpoint(saved, config, cache))
        eval_only = copy.deepcopy(config)
        eval_only["reality_memory"]["objective"]["paired"]["probe_noise_seeds"] = [1, 2, 3]
        eval_only["reality_memory"]["objective"]["paired"]["diagnostic_interval"] = 99
        eval_only["reality_memory"]["references"]["allow_legacy_offline_manifest"] = False
        self.assertTrue(probe.verify_checkpoint(saved, eval_only, cache))
        semantic = copy.deepcopy(config)
        semantic["reality_memory"]["objective"]["paired"]["contrast_weight"] = 0.5
        with self.assertRaises(ValueError):
            probe.verify_checkpoint(saved, semantic, cache)
        broken = copy.deepcopy(saved)
        broken["signature"]["encoder"] = {"encoder": "other"}
        with self.assertRaises(ValueError):
            probe.verify_checkpoint(broken, config, cache)
        with self.assertRaises(ValueError):
            probe.verify_checkpoint({**saved, "stage": "r1"}, config, cache)

    def test_sequential_video_gradients_equal_joint_objective(self):
        sys.path.insert(0, str(ROOT / "code/LongLive"))
        from utils.scheduler import FlowMatchScheduler
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(1000, training=True)
        seen = []
        class Frozen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.4), requires_grad=False)
                self.model = SimpleNamespace(block_mask=None)
            def forward(self, noisy, cond, times, clean_x):
                seen.append((noisy.detach().clone(), times.clone(), clean_x.detach().clone()))
                context = cond["prompt_embeds"].sin().mean()
                flow = self.weight * ((clean_x.cumsum(1) - clean_x) + context)
                return flow, noisy - flow
        frozen = Frozen()
        pipe = SimpleNamespace(generator=frozen, scheduler=scheduler, num_frame_per_block=3, frame_seq_length=1)
        gt = torch.randn(1, 12, 4, 2, 2)
        cond = {"prompt_embeds": torch.randn(1, 5, 12)}
        batch = {"correct_features": torch.randn(1, 2, 3, 6), "wrong_features": torch.randn(1, 2, 3, 6)}
        for initialized in (False, True):
            model = RealityMemory(6, 8, 12, 2)
            if initialized:
                torch.nn.init.normal_(model.output.weight, std=.05)
            direct = copy.deepcopy(model)
            loss, _ = paired_loss(pipe, model, gt, cond, 5, batch, torch.Generator().manual_seed(9), self.config)
            loss.backward()
            # Independent full joint graph, with the same seeds and numerical loss.
            seeds = torch.randint(0, 2**31, (2,), generator=torch.Generator().manual_seed(9)).tolist()
            history = paired_history(gt, 5, self.cfg, "cpu", seeds[0])
            features = torch.cat((batch["correct_features"], batch["wrong_features"]))
            fused, stats = direct(cond["prompt_embeds"].expand(2, -1, -1), features, torch.ones(2, 2, dtype=torch.bool))
            videos = [future_loss(pipe, PreserveHistory(), gt, history, history[:, -1:],
                                  {"prompt_embeds": fused[i:i+1]}, 5, torch.Generator().manual_seed(seeds[1]), 0)
                      for i in (0, 1)]
            joint = torch.stack(videos).mean() + self.cfg["contrast_weight"] * contrast_loss(*stats["relevance_score"], self.cfg["temperature"])
            joint = joint + self.config["reality_memory"]["regularization"]["delta_weight"] * stats["delta_square"]
            joint.backward()
            torch.testing.assert_close(loss, joint)
            for param, reference in zip(model.parameters(), direct.parameters()):
                torch.testing.assert_close(param.grad, reference.grad, atol=1e-7, rtol=1e-5)
            self.assertIsNone(frozen.weight.grad)
            for a, b in zip(seen[-4], seen[-3]):
                self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(seen[-1][2][:, 6:], gt[:, 6:]))

    def test_update_budget_counts_empty_batches_and_resumed_updates(self):
        budget = TrainingBudget(max_updates=3)
        batch, updates = 5, 1  # Resumed counters are totals, not a new budget.
        for active in (False, True, False, True):
            self.assertFalse(budget.done(batch, updates))
            batch, updates = batch + 1, updates + int(active)
        self.assertEqual((batch, updates), (9, 3))
        self.assertTrue(budget.done(batch, updates))
        self.assertTrue(budget.due(50, batch, updates, True))  # Final save even off interval.
        self.assertFalse(TrainingBudget(max_updates=10).due(2, 7, 2, False))
        self.assertTrue(TrainingBudget(max_steps=10).done(10, 6))
        for kwargs in ({}, {"max_steps": 3, "max_updates": 2}, {"max_updates": 0}):
            with self.assertRaises(ValueError):
                TrainingBudget(**kwargs)

    def test_reject_one_sided_gate_penalty_or_empty_probe(self):
        for section, key, value in (("regularization", "wrong_gate_weight", .01),
                                    ("references", "per_reference_dropout", .2)):
            config = copy.deepcopy(self.config)
            config["reality_memory"][section][key] = value
            with self.assertRaises(ValueError):
                validate_paired_config(config)
        self.cfg["probe_noise_seeds"] = []
        with self.assertRaises(ValueError):
            validate_paired_config(self.config)


if __name__ == "__main__":
    unittest.main()
