import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experiments.wan_said.alignment import (Grounding, HiddenCapture, JointGate, Projector,
                                            distillation_loss, masked_pool)
from experiments.wan_said.data import assign_groups, audit_splits, load_split
from experiments.wan_said.io_utils import sha256, write_json, write_jsonl


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.ground = Grounding(12, 10, 8)
        self.text, self.video = torch.randn(4, 12), torch.randn(4, 10)

    def test_mask_pool_padding_dtype(self):
        x = torch.tensor([[[2., 4.], [4., 6.], [999., 999.]]], dtype=torch.bfloat16)
        result = masked_pool(x, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(result, torch.tensor([[3., 5.]]))
        with self.assertRaises(ValueError):
            masked_pool(x, torch.zeros(1, 3))

    def test_mask_shapes_range_text_condition(self):
        t, v = self.ground.encode(self.text, self.video)
        mask, soft = self.ground.gate(t, v)
        self.assertEqual(mask.shape, (4, 8))
        self.assertTrue(((mask > 0) & (mask < 1)).all())
        shuffled, _ = self.ground.gate(t.roll(1, 0), v)
        self.assertGreater((mask - shuffled).abs().mean().item(), 1e-5)
        gate = JointGate(8, hard=True)
        hard, _ = gate(t, v)
        self.assertTrue(((hard == 0) | (hard == 1)).all())
        hard.sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in gate.parameters()))

    def test_consistent_positive_negative_pair_mask(self):
        scores, _ = self.ground.scores(self.text, self.video)
        t, v = self.ground.encode(self.text, self.video)
        for i in range(4):
            for j in range(4):
                m, _ = self.ground.gate(t[j], v[i])
                expected = torch.nn.functional.cosine_similarity(m * v[i], t[j], dim=0) / .07
                torch.testing.assert_close(scores[i, j], expected)

    def test_grounding_backward_parameter_set_teacher_frozen(self):
        teacher = nn.Linear(10, 10).requires_grad_(False)
        video = teacher(self.video)
        loss, _ = self.ground.loss(self.text, video, list('abcd'), list('ABCD'))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        for module in (self.ground.text, self.ground.video, self.ground.gate):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_grounding_no_negative_rejected(self):
        with self.assertRaises(ValueError):
            self.ground.loss(self.text[:1], self.video[:1], ['a'], ['a'])
        with self.assertRaises(ValueError):
            self.ground.loss(self.text, self.video, ['same'] * 4, list('abcd'))
        loss, _ = self.ground.loss(self.text, self.video, ['a', 'a', 'b', 'c'], list('abcd'))
        self.assertTrue(torch.isfinite(loss))

    def test_distill_stop_gradient_zero_mask(self):
        student = torch.randn(4, 8, requires_grad=True)
        target = torch.randn(4, 8, requires_grad=True)
        distillation_loss(student, target).backward()
        self.assertIsNone(target.grad)
        self.assertGreater(student.grad.norm().item(), 0)
        zero = distillation_loss(student, torch.zeros_like(target))
        self.assertEqual(zero.item(), 0)
        zero.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        frozen_target = self.ground.target(self.text, self.video)
        self.assertFalse(frozen_target.requires_grad)

    def test_h15_gradient_checkpoint_capture(self):
        block = nn.Linear(10, 10)
        projection = Projector(10, 8)
        inputs = torch.randn(2, 3, 10)
        with HiddenCapture(block, torch.tensor([[1, 1, 0], [1, 0, 0]])) as capture:
            output = checkpoint(block, inputs, use_reentrant=False)
        loss = distillation_loss(projection(capture.hidden), torch.randn(2, 8))
        loss.backward()
        self.assertGreater(block.weight.grad.norm().item(), 0)
        self.assertGreater(projection[1].weight.grad.norm().item(), 0)
        self.assertEqual(len(block._forward_hooks), 0)
        self.assertEqual(output.shape, (2, 3, 10))

    def test_reentrant_checkpoint_rejected(self):
        with self.assertRaises(RuntimeError):
            with HiddenCapture(nn.Identity()) as capture:
                capture_module = nn.Identity()
                # Detached first passes cannot be used as distillation features.
                capture._capture(capture_module, (), torch.randn(2, 3, 4))

    def test_fm_and_distill_separate_gradient_sets(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(10, 10).requires_grad_(False)
                self.lora_a = nn.Linear(10, 2, bias=False)
                self.lora_b = nn.Linear(2, 10, bias=False)

            def forward(self, x):
                return x + self.base(x) + self.lora_b(self.lora_a(x))

        blocks = nn.ModuleList([Block() for _ in range(30)])
        head = Projector(10, 8)
        for objective in ('fm', 'distill'):
            blocks.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            x = torch.randn(2, 3, 10)
            with HiddenCapture(blocks[14]) as capture:
                for block in blocks:
                    x = checkpoint(block, x, use_reentrant=False)
            loss = x.square().mean() if objective == 'fm' else distillation_loss(head(capture.hidden), torch.randn(2, 8))
            loss.backward()
            active = [i for i, b in enumerate(blocks) if b.lora_b.weight.grad is not None]
            self.assertEqual(active, list(range(30 if objective == 'fm' else 15)))
            self.assertTrue(all(b.base.weight.grad is None for b in blocks))
            self.assertEqual(any(p.grad is not None for p in head.parameters()), objective == 'distill')


class SplitTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [dict(source_id=str(i), sha256=str(i), caption_group=f'c{i}', video=f'/v{i}',
                     previous_split='train' if i < 2 else 'val') for i in range(6)]

    def test_source_hash_caption_group_dedup_and_determinism(self):
        for field in ('source_id', 'sha256', 'caption_group', 'video'):
            rows = self.rows()
            rows[2][field] = rows[0][field]
            assigned, _ = assign_groups(rows)
            self.assertEqual(assigned[2]['split'], 'train')
            audit_splits({s: [r for r in assigned if r['split'] == s] for s in ('train', 'val', 'test')})
        a, _ = assign_groups(self.rows())
        b, _ = assign_groups(list(reversed(self.rows())))
        self.assertEqual({r['source_id']: r['split'] for r in a}, {r['source_id']: r['split'] for r in b})

    def test_split_lock_and_test_optimizer_guard(self):
        rows, _ = assign_groups(self.rows())
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for split in ('train', 'val', 'test'):
                write_jsonl(directory / f'{split}.jsonl', [r for r in rows if r['split'] == split])
            write_json(directory / 'split_lock.json', {'manifest_sha256':
                       {s: sha256(directory / f'{s}.jsonl') for s in ('train', 'val', 'test')}})
            self.assertEqual(len(load_split(directory, 'train', training=True)), 2)
            with self.assertRaises(ValueError):
                load_split(directory, 'test', training=True)
            (directory / 'test.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):
                load_split(directory, 'train', training=True)


if __name__ == '__main__':
    unittest.main()
