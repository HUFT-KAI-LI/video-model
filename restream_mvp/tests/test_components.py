import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import av
import numpy as np
import torch
from restream.anchor_adapter import GatedAnchorAdapter
from restream.anchor_injector import HardAnchorInjector, timestamp_to_latent_index
from restream.corruption import corrupt_history
from restream.dataset import VideoDataset, read_manifest
from restream.metrics import future_errors

ROOT = Path(__file__).resolve().parents[1]


class Components(unittest.TestCase):
    def test_parallel_download_budget_and_oversized_response(self):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace
        import io
        spec = importlib.util.spec_from_file_location("download_youku", ROOT / "scripts/03_download_youku_subset.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        budget = module.ByteBudget(10)
        with ThreadPoolExecutor(max_workers=8) as pool:
            accepted = list(pool.map(budget.reserve, [3] * 30))
        self.assertEqual(sum(accepted), 3)
        self.assertLessEqual(budget.allocated, budget.limit)
        with tempfile.TemporaryDirectory() as directory:
            quota = module.ByteBudget(100)
            bucket = SimpleNamespace(object_exists=lambda key: True,
                                     head_object=lambda key: SimpleNamespace(content_length=4),
                                     get_object=lambda key: io.BytesIO(b"longer than advertised"))
            oss = SimpleNamespace(bucket=bucket, oss_dir="primary", oss_backup_dir="backup")
            row = module.download_one({"video_id:FILE": "videos/example.mp4", "golden_caption": "person"},
                                      oss, Path(directory), quota, 4)
            self.assertEqual(row["error_type"], "ValueError")
            self.assertEqual(quota.allocated, 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_adapter_gradient_bf16_and_reload(self):
        torch.manual_seed(2)
        model = GatedAnchorAdapter(4).bfloat16()
        pred, real = torch.randn(2, 1, 4, 4, 6).bfloat16(), torch.randn(2, 1, 4, 4, 6).bfloat16()
        output = model(pred, real)
        self.assertEqual(output.shape, pred.shape)
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
        self.assertTrue(torch.equal(output, pred))
        self.assertGreater(model.net[-1].weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "adapter.pt"
            torch.save(model.state_dict(), path)
            restored = GatedAnchorAdapter(4).bfloat16()
            restored.load_state_dict(torch.load(path, weights_only=True))
            torch.testing.assert_close(restored(pred, real), output)

    def test_injector_rebuild_and_immutable_input(self):
        x, real = torch.zeros(1, 6, 4, 2, 2), torch.ones(1, 1, 4, 2, 2)
        histories = []
        def rebuild(value):
            histories.append(value.clone())
            return {"kv": value.sum().item()}
        y, state = HardAnchorInjector().inject(x, real, 2, rebuild)
        torch.testing.assert_close(y[:, 2:3], real)
        self.assertEqual(x.sum(), 0)
        self.assertEqual(len(histories), 1)
        self.assertEqual(state["kv"], real.sum().item())
        with self.assertRaises(ValueError):
            HardAnchorInjector().inject(x, real, 8, rebuild)

    def test_time_mapping_is_causal(self):
        # 57 pixels -> 15 latents; each AR block has three latents.
        for requested in np.linspace(0, 5.5, 51):
            index, actual = timestamp_to_latent_index(float(requested), 8., 57)
            self.assertGreaterEqual(actual + 1e-8, requested)
            self.assertEqual((index + 1) % 3, 0)
            self.assertLess(index, 14)
        self.assertEqual(timestamp_to_latent_index(8 * 20 / 56, 8., 57)[0], 5)
        with self.assertRaises(ValueError):
            timestamp_to_latent_index(8, 8, 57)
        with self.assertRaises(ValueError):
            timestamp_to_latent_index(-1, 8, 57)

    def test_corruption_reproducible_and_no_mutation(self):
        x = torch.ones(1, 6, 4, 3, 4)
        a = corrupt_history(x, torch.Generator().manual_seed(3), probability=1)
        b = corrupt_history(x, torch.Generator().manual_seed(3), probability=1)
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(x, torch.ones_like(x))
        self.assertFalse(torch.equal(a, x))
        for prefix in (2, 3, 6):
            result = corrupt_history(x, torch.Generator().manual_seed(3), probability=1, protected_prefix=prefix)
            self.assertTrue(torch.equal(result[:, :prefix], x[:, :prefix]))
            if prefix < x.shape[1]:
                self.assertFalse(torch.equal(result[:, prefix:], x[:, prefix:]))
        for invalid in (-1, 2.5):
            with self.assertRaises(ValueError):
                corrupt_history(x, torch.Generator(), protected_prefix=invalid)

    def test_future_metrics_exclude_anchor_and_prefix(self):
        target = torch.zeros(1, 15, 4, 2, 2)
        pred = target.clone()
        pred[:, :6] = 100
        metrics = future_errors(pred, target, 5, 8)
        self.assertEqual(metrics["future_latent_mse"], 0)
        self.assertIsNone(metrics["future_latent_mse_0.5s"])

    def test_prepare_uses_configured_sink_prefix(self):
        from types import SimpleNamespace
        from restream.objective import prepare
        vae = SimpleNamespace(encode_to_latent=lambda pixels: pixels[:, :, ::4].mean((3, 4), keepdim=True).permute(0, 2, 1, 3, 4))
        pipe = SimpleNamespace(vae=vae, num_frame_per_block=3, text_encoder=lambda captions: {})
        batch = {"pixels": torch.zeros(1, 3, 57, 16, 16), "anchor_sec": [1.2],
                 "window_sec": [3.5], "caption": ["fixture"]}
        for prefix in (2, 6):
            config = {"reanchor": {"drift_probability": 1., "protected_sink_latents": prefix}}
            gt, history, _, _, index, _ = prepare(pipe, batch, "cpu", config, torch.Generator().manual_seed(3))
            self.assertEqual(index, 5)
            self.assertTrue(torch.equal(history[:, :prefix], gt[:, :prefix]))
            if prefix == 2:
                self.assertFalse(torch.equal(history[:, prefix:], gt[:, prefix:6]))

    def test_dataset_split_decode_and_normalization(self):
        spec = importlib.util.spec_from_file_location("build_manifest", ROOT / "scripts/04_build_manifest.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video = folder / "fixture.mp4"
            with av.open(str(video), "w") as c:
                s = c.add_stream("mpeg4", rate=16)
                s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
                for i in range(80):
                    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
                    pixels[:, :, 0] = 200
                    pixels[:, :, 2] = i * 3
                    for packet in s.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        c.mux(packet)
                for packet in s.encode():
                    c.mux(packet)
            raw = folder / "raw.jsonl"
            # Synthetic IDs exercise grouping logic only, never constitute downloaded data.
            raw.write_text("".join(json.dumps({"video": str(video), "source_id": str(i // 2),
                                                "duration": 5., "caption": "fixture"}) + "\n" for i in range(40)))
            module.build(raw, folder / "data")
            train = read_manifest(folder / "data/train.jsonl")
            val = read_manifest(folder / "data/val.jsonl")
            self.assertFalse({r["source_id"] for r in train} & {r["source_id"] for r in val})
            dataset = VideoDataset(folder / "data/train.jsonl", 57, 32, 48, 16)
            sample = dataset[0]
            self.assertEqual(sample["pixels"].shape, (3, 57, 32, 48))
            self.assertTrue(torch.isfinite(sample["pixels"]).all())
            self.assertGreater(sample["pixels"][0].mean(), .4)
            self.assertLess(sample["pixels"][1].mean(), -.9)
            self.assertGreater(sample["anchor_sec"], 0)
            self.assertLess(sample["anchor_sec"], sample["window_sec"])
            with self.assertRaisesRegex(ValueError, "rebuild the manifest"):
                VideoDataset(folder / "data/train.jsonl", 57, 32, 48, 8)
            # The same 33-frame request covers 2s at 16 FPS and 4s at 8 FPS.
            samples = []
            for fps in (16, 8):
                path = folder / f"fps_{fps}.jsonl"
                path.write_text(json.dumps({**train[0], "window_start": 0, "window_sec": 32 / fps}) + "\n")
                samples.append(VideoDataset(path, 33, 32, 48, fps)[0]["pixels"])
            self.assertGreater((samples[1][2, -1] - samples[0][2, -1]).mean().item(), .5)
            module.build(raw, folder / "slow", frames=33, fps=8)
            self.assertTrue(all(r["window_sec"] == 4 for r in read_manifest(folder / "slow/train.jsonl")))

    def test_future_objective_gradient_through_frozen_backbone(self):
        from types import SimpleNamespace
        from restream.objective import future_loss
        sys.path.insert(0, str(ROOT / "code/LongLive"))
        from utils.scheduler import FlowMatchScheduler
        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(1000, training=True)
        class Frozen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.7), requires_grad=False)
                self.model = SimpleNamespace(block_mask=None)
                self.late_error = 0
                self.seen_masks = []
            def forward(self, noisy, cond, timestep, clean_x):
                # Causal proxy: a future output depends on earlier clean context.
                context = torch.cumsum(clean_x, dim=1) - clean_x
                flow = self.weight * context
                flow[:, 9:] += self.late_error
                self.seen_masks.append(self.model.block_mask)
                self.model.block_mask = "cached"
                return flow, noisy - flow
        backbone = Frozen()
        pipe = SimpleNamespace(generator=backbone, scheduler=scheduler, num_frame_per_block=3, frame_seq_length=1)
        adapter = GatedAnchorAdapter(4)
        gt = torch.randn(1, 12, 4, 2, 2)
        history = gt[:, :6] + .1
        # reg=0 proves gradients originate in FUTURE supervision, not delta regularizer.
        loss = future_loss(pipe, adapter, gt, history, gt[:, 5:6], {}, 5,
                           torch.Generator().manual_seed(9), 0)
        loss.backward()
        self.assertGreater(adapter.net[-1].weight.grad.abs().sum().item(), 0)
        self.assertEqual(adapter.gate_logit.grad.item(), 0)  # Zero output layer at init.
        self.assertIsNone(backbone.weight.grad)
        backbone.late_error = 1000
        same = future_loss(pipe, adapter, gt, history, gt[:, 5:6], {}, 5,
                           torch.Generator().manual_seed(9), 0)
        torch.testing.assert_close(same, loss)  # Blocks after the first future block are excluded.
        wider = torch.randn(1, 12, 4, 2, 4)
        pipe.frame_seq_length = 2
        future_loss(pipe, adapter, wider, wider[:, :6], wider[:, 5:6], {}, 5,
                    torch.Generator().manual_seed(9), 0)
        self.assertEqual(backbone.seen_masks, [None, "cached", None])


if __name__ == "__main__":
    unittest.main()
