import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import av
import numpy as np
import torch
from restream.reality_cache import FeatureCache
from restream.reality_dataset import collate_reality, RealityDataset
from restream.reality_memory import RealityMemory, reference_dropout, memory_regularization
from restream.reality_metrics import visual_reference_metrics, summarize_cases
from restream.reality_runtime import read_reality_config

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RealityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = RealityMemory(6, 8, 12, 2)
        self.context = torch.randn(2, 5, 12)
        self.context[:, -1] = 0

    def test_zero_init_variable_reference_count_and_future_gradient(self):
        for k in (0, 1, 2, 4, 8):
            self.model.zero_grad(set_to_none=True)
            features = torch.randn(2, k, 3, 6)
            output, stats = self.model(self.context, features, torch.ones(2, k, dtype=torch.bool))
            self.assertTrue(torch.equal(output, self.context))
            output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))
            self.assertEqual(stats["gate"].shape, (2,))
            if k:
                self.assertGreater(self.model.output.weight.grad.abs().sum().item(), 0)
            else:
                self.assertEqual(sum(p.grad.abs().sum().item() for p in self.model.parameters()), 0)

    def test_no_memory_exact_even_after_output_learns(self):
        torch.nn.init.normal_(self.model.output.weight)
        torch.nn.init.normal_(self.model.output.bias)
        features = torch.randn(2, 4, 3, 6)
        features[0] = float("nan")  # Invalid padded evidence is ignored structurally.
        mask = torch.tensor([[False] * 4, [True, False, True, False]])
        result, stats = self.model(self.context, features, mask)
        self.assertTrue(torch.equal(result[0], self.context[0]))
        self.assertFalse(torch.equal(result[1], self.context[1]))
        self.assertTrue(torch.equal(result[:, -1], self.context[:, -1]))
        self.assertEqual(stats["gate"][0].item(), 0)
        self.assertTrue(torch.isfinite(result).all())

    def test_reference_order_invariance_but_donor_changes_context(self):
        torch.nn.init.normal_(self.model.output.weight, std=.05)
        features = torch.randn(2, 4, 3, 6)
        mask = torch.ones(2, 4, dtype=torch.bool)
        original, _ = self.model(self.context, features, mask)
        reordered, _ = self.model(self.context, features.flip(1), mask)
        swapped, _ = self.model(self.context, features.flip(0), mask)
        torch.testing.assert_close(original, reordered)
        self.assertFalse(torch.allclose(original, swapped))

    def test_wrong_gate_supervision_uses_labels_only_in_loss(self):
        _, stats = self.model(self.context, torch.randn(2, 2, 3, 6), torch.ones(2, 2, dtype=torch.bool))
        penalty, wrong = memory_regularization(stats, torch.tensor([False, True]), 0, 1)
        torch.testing.assert_close(wrong, stats["gate"][1].square())
        gradient = torch.autograd.grad(penalty, stats["gate"], retain_graph=True)[0]
        self.assertEqual(gradient[0].item(), 0)
        self.assertGreater(gradient[1].item(), 0)
        stats["active"] = torch.tensor([True, False])
        _, wrong = memory_regularization(stats, torch.tensor([False, True]), 0, 1)
        self.assertEqual(wrong.item(), 0)

    def test_reference_dropout_reproducible_and_does_not_restore_padding(self):
        mask = torch.tensor([[True, False] * 20])
        first = reference_dropout(mask, torch.Generator().manual_seed(5), .5)
        second = reference_dropout(mask, torch.Generator().manual_seed(5), .5)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first & ~mask, torch.zeros_like(mask)))
        self.assertTrue(0 < first.sum() < mask.sum())
        self.assertTrue(torch.equal(reference_dropout(mask, torch.Generator(), 1, False), mask))
        self.assertFalse(reference_dropout(mask, torch.Generator(), 1).any())

    def test_normalized_entropy_uses_valid_tokens_and_handles_empty(self):
        for parameter in self.model.query.parameters():
            torch.nn.init.zeros_(parameter)
        for count in (1, 2, 4, 8):
            mask = torch.zeros(2, 8, dtype=torch.bool)
            mask[1, :count] = True
            _, stats = self.model(self.context, torch.randn(2, 8, 3, 6), mask)
            self.assertEqual(stats["valid_memory_tokens"].tolist(), [0, count * 3])
            torch.testing.assert_close(stats["attention_entropy_normalized"], torch.tensor([0., 1.]))
        _, stats = self.model(self.context, torch.randn(2, 1, 1, 6), torch.ones(2, 1, dtype=torch.bool))
        self.assertEqual(stats["attention_entropy_normalized"].tolist(), [0., 0.])

    def test_target_seek_matches_sequential_decode(self):
        from restream.dataset import VideoDataset
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "seek.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=8)
                stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
                stream.codec_context.gop_size = 12
                for i in range(96):
                    pixels = np.full((48, 64, 3), (i * 2, 64, 255 - i * 2), dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            manifest = Path(folder) / "samples.jsonl"
            manifest.write_text(json.dumps({"video": str(path), "window_start": 7.3, "window_sec": 2.,
                                            "caption": "fixture", "source_id": "fixture", "anchor_sec": [1.]}) + "\n")
            seek = VideoDataset(manifest, 33, 32, 48, 16, seek=True)[0]
            sequential = VideoDataset(manifest, 33, 32, 48, 16, seek=False)[0]
            self.assertTrue(torch.equal(seek["pixels"], sequential["pixels"]))

    def test_feature_cache_reload_and_stale_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = FeatureCache(folder, {"weights": "v1"}, 3, 6)
            ref = {"video_sha256": "video_a", "time": 1.25}
            features = torch.randn(3, 6)
            cache.write(ref, features)
            torch.testing.assert_close(cache.read(ref), features, rtol=0, atol=0)
            with self.assertRaises(FileNotFoundError):
                FeatureCache(folder, {"weights": "v2"}, 3, 6).read(ref)
            with self.assertRaises(FileNotFoundError):
                cache.read({**ref, "time": 2.5})
            torch.save({"key": "bad", "features": features}, cache.path(ref))
            with self.assertRaises(ValueError):
                cache.read(ref)

    def test_collation_no_reference_and_variable_k(self):
        samples = [{"features": torch.ones(k, 3, 6), "reference_mask": torch.ones(k, dtype=torch.bool),
                    "wrong_reference": False, "caption": "fixture"} for k in (0, 1, 4)]
        batch = collate_reality(samples)
        self.assertEqual(batch["features"].shape, (3, 4, 3, 6))
        self.assertEqual(batch["reference_mask"].sum(1).tolist(), [0, 1, 4])
        single = collate_reality(samples[:1])
        self.assertEqual(single["features"].shape, (1, 1, 3, 6))
        self.assertFalse(single["reference_mask"].any())

    def test_manifest_async_gap_split_and_mixture(self):
        module = script("build_reality_manifest")
        config = read_reality_config(ROOT / "configs/reality_memory_r0.yaml")
        row = {"source_id": "source", "video": "fixture.mp4", "sha256": "a", "split": "train", "caption": "fixture"}
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        def decode(ref, size, return_time=False):
            return (rgb, ref["time"]) if return_time else rgb
        with patch.object(module, "read_reference", side_effect=decode):
            item = module.candidate(row, [{"start": 0, "end": 30}], config)
        self.assertIsNotNone(item)
        self.assertEqual(len(item["reference_sets"]["async"]), 8)
        for ref in item["reference_sets"]["async"]:
            self.assertLessEqual(ref["time"], item["target_start"] - 1.5 + 1e-6)
        for split in ("train", "val"):
            rows = []
            for index in range(20):
                clone = copy.deepcopy(item)
                clone.update(source_id=f"{split}_{index}", split=split)
                for refs in clone["reference_sets"].values():
                    for ref in refs:
                        ref.update(source_id=clone["source_id"], split=split)
                rows.append(clone)
            module.attach_references(rows, config)
            counts = {kind: sum(r["reference_kind"] == kind for r in rows) for kind in ("async", "aligned", "none", "wrong")}
            self.assertEqual(counts, {"async": 10, "aligned": 4, "none": 3, "wrong": 3})
            for record in rows:
                for ref in record["references"]:
                    self.assertEqual(ref["split"], split)
                    self.assertEqual(ref["source_id"] != record["source_id"], record["reference_kind"] == "wrong")

    def test_shot_filter_rejects_black_and_splits_cuts(self):
        module = script("filter_continuous_shots")
        config = read_reality_config(ROOT / "configs/reality_memory_r0.yaml")["reality_memory"]["filter"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cuts.mp4"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("mpeg4", rate=8)
                stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
                for color in [(240, 0, 0)] * 40 + [(0, 0, 240)] * 40 + [(0, 0, 0)] * 8 + [(0, 240, 0)] * 40:
                    pixels = np.full((64, 64, 3), color, dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            result = module.scan({"video": str(path), "source_id": "x", "sha256": "x"}, config)
            self.assertIsNone(result["error"])
            self.assertGreaterEqual(result["hard_cuts"], 2)
            self.assertEqual(result["black_frames"], 8)
            self.assertGreaterEqual(len(result["shots"]), 3)
            for shot in result["shots"]:
                self.assertFalse(shot["start"] < 10 and shot["end"] > 10)

    def test_reference_metrics_empty_and_matching(self):
        generated = torch.randn(4, 3, 6)
        self.assertIsNone(visual_reference_metrics(generated, torch.empty(0, 3, 6))["reference_copy_score"])
        result = visual_reference_metrics(generated, generated.clone())
        self.assertAlmostEqual(result["reference_copy_score"], 1, places=5)
        summary = summarize_cases([{"variants": {"async_k1": {"future_latent_mse": 1.}, "wrong_k1": {"future_latent_mse": 2.}}}])
        self.assertEqual(summary["correct_vs_wrong_gap"]["async_k1"], 1.)

    def test_r1_config_is_rejected(self):
        config = read_reality_config(ROOT / "configs/reality_memory_r0.yaml")
        config["reality_memory"]["stage"] = "r1"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "r1.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Only R0"):
                read_reality_config(path)

    def test_evaluation_controls_preserve_base_and_restore_training_mode(self):
        import eval_reality_memory as evaluation
        from types import SimpleNamespace
        config = read_reality_config(ROOT / "configs/reality_memory_r0.yaml")
        gt = torch.randn(1, 15, 4, 2, 2)
        context = torch.randn(1, 5, 12)
        memory = RealityMemory(6, 8, 12, 2)
        memory.train()
        pools = {kind: [{"feature": torch.randn(3, 6)} for _ in range(2)] for kind in ("async", "aligned", "wrong")}
        class Dataset:
            cache = SimpleNamespace(tokens=3, channels=6)
            rows = [{"sample_id": str(i), "source_id": str(i), "target_start": 5., "target_sec": 3.5,
                     "reference_sets": pools} for i in range(2)]
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return {"pixels": torch.zeros(3, 57, 16, 16), "features": torch.empty(0, 3, 6),
                        "reference_mask": torch.empty(0, dtype=torch.bool), "caption": "fixture"}
            def reference_features(self, refs):
                return torch.stack([ref["feature"] for ref in refs])
        vae = SimpleNamespace(encode_to_latent=lambda image: torch.zeros(1, 1, 4, 2, 2),
                              decode_to_pixel=lambda latent: torch.zeros(1, 57, 3, 16, 16))
        pipe = SimpleNamespace(vae=vae, num_frame_per_block=3)
        noises = []
        def rollout(pipeline, prefix, cond, noise, rng):
            noises.append(noise.clone())
            return torch.cat((prefix, noise), 1)
        with tempfile.TemporaryDirectory() as folder, patch.object(evaluation, "make_dataset", return_value=Dataset()), \
                patch.object(evaluation, "prepare_reality", return_value=(gt, {"prompt_embeds": context}, 5)), \
                patch.object(evaluation, "rollout", side_effect=rollout), patch.object(evaluation, "write_video"):
            result = evaluation.evaluate(pipe, memory, config, "cpu", Path(folder), cases=1, counts=[0, 1])
        self.assertTrue(memory.training)
        self.assertEqual(len(result["cases"][0]["variants"]), 8)
        self.assertEqual(result["cases"][0]["variants"]["no_memory"]["max_abs_difference_from_base"], 0)
        self.assertTrue(all(torch.equal(noise, noises[0]) for noise in noises))


if __name__ == "__main__":
    unittest.main()
