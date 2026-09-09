import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from restream.objective import future_loss
from restream.reality_memory import RealityMemory
from restream.reality_paired import (paired_loss, paired_history, paired_references,
                                    contrast_loss, validate_paired_config)
from restream.reality_runtime import PreserveHistory, read_reality_config
from restream.training_budget import TrainingBudget

ROOT = Path(__file__).resolve().parents[1]


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
