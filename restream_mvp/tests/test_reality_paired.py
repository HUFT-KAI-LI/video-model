import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import tempfile
import numpy as np
import torch
from restream.objective import future_loss
from restream.reality_memory import RealityMemory
from restream.reality_dataset import RealityDataset
from restream.reality_paired import (paired_loss, paired_history, paired_references,
                                    contrast_loss, validate_paired_config, load_global_constant,
                                    global_constant_provenance)
from restream.reality_paired_diagnostics import aggregate_pairs
from restream.reality_runtime import PreserveHistory, read_reality_config
from restream.training_budget import TrainingBudget
from restream.reality_selection import GLOBAL_MEAN_SCHEMA, selection_config_hash
from restream.reality_data import canonical_hash

ROOT = Path(__file__).resolve().parents[1]


def minimal_paired_config(train_manifest=None, protocol="offline_target_filtered", global_path=None):
    """Complete config for selection identity and global-mean provenance checks."""
    return {
        "seed": 7,
        "data": {"frames": 57, "fps": 16.0,
                 "train_manifest": str(train_manifest or (ROOT / "data/reality_train.jsonl"))},
        "eval": {"reference_counts": [1, 2]},
        "reality_memory": {
            "references": {"selection_protocol": protocol, "async_direction": "past_only",
                           "min_gap_sec": 1.5, "boundary_margin_sec": 0.2,
                           "near_radius_sec": 0.5, "max_count": 2, "min_count": 1,
                           **({"global_constant_features": str(global_path)} if global_path else {})},
            "filter": {"scene_similarity": 0.65},
            "objective": {"prefix_latents": 6},
        },
    }


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
                    "correct": {"video_loss": .6 * value_scale, "relevance_score": 0.5, "memory_gate_mean": .2},
                    "wrong_source": {"video_loss": .95 * value_scale, "relevance_score": 0.4, "memory_gate_mean": .2}}
        cases = []
        for sample_id, scale in (("a", 1.0), ("b", 1.0)):
            for mode in ("clean", "mild"):
                for seed in (1, 2):
                    cases.append({"sample_id": sample_id, "prefix_mode": mode, "noise_seed": seed,
                                  "variants": variants_for(scale)})
        aggregate = aggregate_pairs(cases)["clean"]["core_deltas"]
        for name, expected in (("U_correct", .4), ("G_branch", .1), ("G_generic", .1),
                               ("G_content", .2), ("S_reference", .35)):
            self.assertAlmostEqual(aggregate[name], expected, places=6)
        self.assertEqual(aggregate_pairs(cases)["clean"]["core_delta_targets"]["U_correct"], 2)

    def test_global_constant_is_fixed_and_provenance_bound_to_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            manifest = folder / "train.jsonl"
            identity = {"encoder": "fixture", "version": 1}
            shared = {"video_sha256": "v1", "time": 1.0, "split": "train", "source_id": "a", "role": "async"}
            rows = [{"source_id": "a", "split": "train", "sample_id": "s1",
                     "reference_sets": {"async": [{**shared, "time": 1.0}], "aligned": []}},
                    {"source_id": "a", "split": "train", "sample_id": "s2",
                     "reference_sets": {"async": [{**shared, "time": 1.0}, {**shared, "time": 2.0}], "aligned": []}}]
            manifest.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows))
            cache = SimpleNamespace(identity=identity, tokens=3, channels=4,
                                    key=lambda ref: canonical_hash({"video_sha256": ref["video_sha256"],
                                                                    "time": ref["time"], "encoder": identity}),
                                    read=lambda ref: torch.arange(12, dtype=torch.float32).reshape(3, 4) * (ref["time"] / 3 + 1))
            path = folder / "mean.pt"
            config = minimal_paired_config(train_manifest=manifest, global_path=path)
            mean_value = torch.arange(12, dtype=torch.float32).reshape(3, 4) * 1.5
            torch.save({"schema": GLOBAL_MEAN_SCHEMA,
                        "features": mean_value,
                        "source_split": "train", "unique_reference_count": 2,
                        "cache_identity": identity, "tokens": 3, "channels": 4,
                        "selection_protocol": "offline_target_filtered",
                        "train_manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
                        "reference_keys_sha256": canonical_hash(sorted({cache.key(r) for row in rows for r in row["reference_sets"]["async"]})),
                        "selection_config_hash": selection_config_hash(config)}, path)
            first = load_global_constant(config, cache, "cpu")
            second = load_global_constant(config, cache, "cpu")
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.equal(first, mean_value))
            self.assertEqual(global_constant_provenance(config, cache, "cpu")["unique_reference_count"], 2)
            cache.identity = {"encoder": "other"}
            with self.assertRaises(ValueError):
                load_global_constant(config, cache, "cpu")
            cache.identity = identity
            # Regenerating the train manifest must invalidate the stored control.
            manifest.write_text("".join(__import__("json").dumps({**rows[0],
                                                                  "reference_sets": {"async": [{**shared, "time": 9.0}], "aligned": []}}) + "\n"
                                 for _ in range(1)))
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
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_reality_manifest", ROOT / "scripts/build_reality_manifest.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
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
        visible prefix, that visible_until is the actual boundary frame, and that
        the row carries the shared selection hash."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_reality_manifest", ROOT / "scripts/build_reality_manifest.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        step = 1 / 16.0
        calls = []

        def recording_decoder(reference, size=None, return_time=False, latest=False):
            # Real videos quantize frame starts to a grid; requested times are
            # arbitrary, so the first at/after frame can start after the request.
            index = math.floor(reference["time"] / step + 1e-9) if latest else math.ceil(reference["time"] / step - 1e-9)
            actual = float(index * step)
            calls.append({"time": reference["time"], "actual": actual, "latest": latest, "role": reference.get("role")})
            rgb = np.full((64, 64, 3), 200, dtype=np.uint8)  # Same histogram for every frame.
            return (rgb, actual) if return_time else rgb

        module.read_reference = recording_decoder
        row = {"source_id": "s1", "video": "/tmp/fake.mp4", "sha256": "deadbeef", "split": "train"}
        shots = [{"start": 0.0, "end": 40.0}]
        prefix_latents, fps = 6, 16.0
        config = minimal_paired_config(protocol="strict_online")
        config["reality_memory"]["references"]["max_count"] = 4
        config["eval"]["reference_counts"] = [1, 2, 4]
        item = module.candidate(row, shots, config)
        self.assertIsNotNone(item)
        nominal = item["target_start"] + 4 * (prefix_latents - 1) / fps
        visible = item["visible_until"]
        self.assertIsNotNone(visible)
        self.assertLessEqual(visible, nominal + 1e-6)
        boundary_calls = [c for c in calls if c["role"] == "analysis" and abs(c["time"] - nominal) < 1e-6]
        self.assertTrue(boundary_calls)
        self.assertTrue(all(c["latest"] for c in boundary_calls), "Boundary analysis must read at-or-before")
        for kind in ("async", "aligned"):
            self.assertTrue(item["reference_sets"][kind], f"Expected {kind} references")
            for ref in item["reference_sets"][kind]:
                self.assertLessEqual(ref["time"], visible + 1e-6,
                                     f"strict_online {kind} reference exceeds visible_until")
        self.assertEqual(item["selection_protocol"], "strict_online")
        self.assertEqual(item["selection_config_hash"], selection_config_hash(config))
        self.assertEqual(item["selection_schema"], module.SELECTION_IDENTITY_SCHEMA)

    def test_candidate_offline_still_filters_with_full_target_histogram(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_reality_manifest", ROOT / "scripts/build_reality_manifest.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        calls = []

        def recording_decoder(reference, size=None, return_time=False, latest=False):
            actual = float(reference["time"])
            calls.append({"time": reference["time"], "latest": latest, "role": reference.get("role")})
            rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
            return (rgb, actual) if return_time else rgb

        module.read_reference = recording_decoder
        row = {"source_id": "s1", "video": "/tmp/fake.mp4", "sha256": "deadbeef", "split": "train"}
        shots = [{"start": 0.0, "end": 40.0}]
        config = minimal_paired_config(protocol="offline_target_filtered")
        config["reality_memory"]["references"]["max_count"] = 4
        config["eval"]["reference_counts"] = [1, 2, 4]
        item = module.candidate(row, shots, config)
        self.assertIsNotNone(item)
        self.assertIsNone(item["visible_until"])
        self.assertEqual(item["selection_protocol"], "offline_target_filtered")
        arrival = 4 * (config["reality_memory"]["objective"]["prefix_latents"] - 1) / config["data"]["fps"]
        analysis = [c for c in calls if c["role"] == "analysis"]
        self.assertGreater(max(c["time"] for c in analysis), item["target_start"] + arrival,
                           "offline filtering legitimately reads the full target histogram")
        self.assertEqual(item["selection_config_hash"], selection_config_hash(config))

    def test_global_mean_pool_is_deduplicated_and_mixture_independent(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("cache_reality_features", ROOT / "scripts/cache_reality_features.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        identity = {"encoder": "fixture", "version": 1}
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / "train.jsonl"
            rows = [{"source_id": "a", "split": "train", "sample_id": "s1",
                     "reference_sets": {"async": [{"video_sha256": "v", "time": 1.0}, {"video_sha256": "v", "time": 2.0}],
                                        "aligned": []}},
                    {"source_id": "a", "split": "train", "sample_id": "s2",
                     "reference_sets": {"async": [{"video_sha256": "v", "time": 1.0}],
                                        "aligned": [{"video_sha256": "v", "time": 2.0}]}}]
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
            cache = SimpleNamespace(identity=identity, tokens=3, channels=4,
                                    key=lambda ref: canonical_hash({"video_sha256": ref["video_sha256"],
                                                                    "time": ref["time"], "encoder": identity}),
                                    read=lambda ref: torch.arange(12, dtype=torch.float32).reshape(3, 4) * ref["time"])
            config = minimal_paired_config(train_manifest=manifest)
            payload = module.global_mean_payload(rows, cache, config)
            # Two distinct frames (t=1, t=2), despite appearing four times across rows.
            self.assertEqual(payload["unique_reference_count"], 2)
            torch.testing.assert_close(payload["features"], torch.arange(12, dtype=torch.float32).reshape(3, 4) * 1.5)
            self.assertEqual(payload["schema"], GLOBAL_MEAN_SCHEMA)
            self.assertEqual(payload["source_split"], "train")
            self.assertEqual(payload["train_manifest_sha256"], __import__("hashlib").sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(payload["selection_config_hash"], selection_config_hash(config))

    def test_reality_dataset_enforces_strict_online_bounds_and_selection_hash(self):
        base = {"window_start": 0.0, "window_sec": 3.5, "target_start": 0.0, "target_sec": 3.5,
                "split": "train", "source_id": "a", "reference_kind": "async",
                "references": [{"split": "train", "source_id": "a", "time": .5}],
                "caption": "c", "anchor_sec": [1.25], "video": "/tmp/fake.mp4", "sha256": "x",
                "visual_review": "pending", "filter_status": "pass", "shot": {"start": 0, "end": 5},
                "shot_id": 0, "window_start_sec": 0.0}

        def refs(*times):
            return {"async": [{"split": "train", "source_id": "a", "time": t} for t in times], "aligned": []}

        cache = SimpleNamespace(identity="x", tokens=1, channels=1, path=lambda ref: Path("/nonexistent"))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "m.jsonl"

            def load(rows, protocol=None, selection_hash=None):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                return RealityDataset(path, cache, preflight=False,
                                      selection_protocol=protocol, selection_config_hash=selection_hash)

            good = {**base, "sample_id": "good", "selection_protocol": "strict_online",
                    "selection_schema": 2, "visible_until": 2.0, "reference_sets": refs(.5, 1.5),
                    "selection_config_hash": "expected-hash"}
            load([good], protocol="strict_online", selection_hash="expected-hash")
            with self.assertRaises(ValueError):
                load([{**good, "sample_id": "late", "reference_sets": refs(.5, 2.5)}],
                     protocol="strict_online", selection_hash="expected-hash")
            with self.assertRaises(ValueError):
                load([{**good, "sample_id": "no-visible", "visible_until": None}],
                     protocol="strict_online", selection_hash="expected-hash")
            with self.assertRaises(ValueError):
                load([{**good, "sample_id": "stale-hash", "selection_config_hash": "stale"}],
                     protocol="strict_online", selection_hash="expected-hash")
            with self.assertRaises(ValueError):
                load([{**base, "sample_id": "wrong-protocol", "reference_sets": refs(.5)}],
                     protocol="strict_online", selection_hash="expected-hash")
            # Legacy rows predate the protocol/hash fields and load under the default.
            load([{**base, "sample_id": "legacy", "reference_sets": refs(.5)}], protocol="offline_target_filtered")

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
