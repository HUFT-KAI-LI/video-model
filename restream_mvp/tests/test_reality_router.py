import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_memory import RealityMemory
from restream.reality_router import FrozenStateRouter
from restream.reality_temporal import (assert_visible_frame_indices, latent_prefix_boundary_index,
                                       pixel_count_for_latent_prefix, prefix_visible_seconds)


class TemporalMappingTests(unittest.TestCase):
    def test_wan_grouping_matches_latent_to_pixel_mapping(self):
        """Wan maps frame 0 -> latent 0 and frames 1..4, 5..8, ... -> later latents,
        so L prefix latents expose 4*(L-1)+1 pixel frames."""
        for latents, count, boundary in ((3, 9, 8), (6, 21, 20), (9, 33, 32), (1, 1, 0)):
            self.assertEqual(pixel_count_for_latent_prefix(latents), count)
            self.assertEqual(latent_prefix_boundary_index(latents), boundary)
        self.assertAlmostEqual(prefix_visible_seconds(6, 16), 20 / 16)
        self.assertAlmostEqual(prefix_visible_seconds(6, 8), 2.5)
        for bad in (0, -1, 1.5, None, "6"):
            with self.assertRaises(ValueError):
                pixel_count_for_latent_prefix(bad)
        with self.assertRaises(ValueError):
            prefix_visible_seconds(6, 0)

    def test_retrieval_query_frames_stay_inside_visible_prefix(self):
        """Regression for the 3-frame future leak: 6 latents expose 0..20, so the
        three-frame query must be [0, 10, 20] and [0, 12, 23] must be rejected."""
        self.assertEqual(assert_visible_frame_indices([0, 10, 20], 6), [0, 10, 20])
        with self.assertRaises(ValueError):
            assert_visible_frame_indices([0, 12, 23], 6)
        with self.assertRaises(ValueError):
            assert_visible_frame_indices([0, 21], 6)
        with self.assertRaises(ValueError):
            assert_visible_frame_indices([-1, 0], 6)


class RouterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.tokens, self.dim = 4, 8
        self.prefix = torch.randn(self.tokens, self.dim)
        self.candidates = torch.randn(3, self.tokens, self.dim)

    def test_router_has_no_trainable_parameters(self):
        self.assertEqual(list(FrozenStateRouter().parameters()), [])

    def test_soft_routing_weights_and_weighted_sum(self):
        router = FrozenStateRouter(temperature=0.05, mode="soft")
        aligned = self.prefix.clone()
        candidates = torch.stack([torch.randn(self.tokens, self.dim), aligned, torch.randn(self.tokens, self.dim)])
        memory, mask, stats = router(self.prefix, candidates)
        self.assertEqual(memory.shape, (1, 1, self.tokens, self.dim))
        self.assertEqual(mask.shape, (1, 1))
        self.assertTrue(bool(mask.all()))
        self.assertAlmostEqual(float(stats["router_weights"].sum()), 1.0, places=5)
        self.assertEqual(int(stats["router_weights"].argmax()), 1)
        expected = (stats["router_weights"][..., None, None] * candidates.float()).sum(1, keepdim=True)
        torch.testing.assert_close(memory, expected.to(memory.dtype))

    def test_mask_removes_candidates_from_the_residual(self):
        router = FrozenStateRouter(temperature=0.05, mode="soft")
        _, output_mask, stats = router(self.prefix, self.candidates, torch.tensor([True, False, True]))
        self.assertAlmostEqual(float(stats["router_weights"][0, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(stats["router_weights"].sum()), 1.0, places=5)
        self.assertTrue(bool(output_mask.all()))
        memory, output_mask, stats = router(self.prefix, self.candidates, torch.zeros(3, dtype=torch.bool))
        self.assertFalse(bool(output_mask.any()))
        self.assertFalse(bool(stats["active"].any()))
        self.assertTrue(torch.equal(memory, torch.zeros_like(memory)))

    def test_topk_selects_the_highest_scoring_candidates(self):
        router = FrozenStateRouter(temperature=0.05, mode="topk", top_k=2)
        memory, mask, stats = router(self.prefix, self.candidates)
        self.assertEqual(memory.shape, (1, 2, self.tokens, self.dim))
        expected = torch.topk(stats["router_scores"][0], 2).indices.tolist()
        self.assertEqual(stats["selected_indices"][0].tolist(), expected)
        self.assertAlmostEqual(float(stats["selected_weights"].sum()), 1.0, places=5)
        self.assertTrue(bool(mask.all()))

    def test_routing_is_structurally_coupled_to_generation(self):
        memory_model = RealityMemory(self.dim, 6, 12, 2)
        torch.nn.init.normal_(memory_model.output.weight, std=.05)
        torch.nn.init.normal_(memory_model.output.bias, std=.05)
        context = torch.randn(1, 3, 12)
        router = FrozenStateRouter(temperature=0.1, mode="soft")
        first_memory, first_mask, first_stats = router(self.prefix, self.candidates)
        second_memory, second_mask, second_stats = router(torch.randn(self.tokens, self.dim), self.candidates)
        self.assertFalse(torch.allclose(first_memory, second_memory))
        self.assertFalse(torch.allclose(first_stats["router_weights"], second_stats["router_weights"]))
        fused_first, _ = memory_model(context, first_memory, first_mask)
        fused_second, _ = memory_model(context, second_memory, second_mask)
        self.assertEqual(fused_first.shape, context.shape)
        self.assertFalse(torch.allclose(fused_first, fused_second))
        topk = FrozenStateRouter(mode="topk", top_k=2)
        fused_topk, _ = memory_model(context, *topk(self.prefix, self.candidates)[:2])
        self.assertEqual(fused_topk.shape, context.shape)

    def test_invalid_router_inputs_raise(self):
        router = FrozenStateRouter()
        for kwargs in ({"mode": "bad"}, {"temperature": 0}, {"mode": "topk", "top_k": 0}):
            with self.assertRaises(ValueError):
                FrozenStateRouter(**kwargs)
        with self.assertRaises(ValueError):
            router(self.prefix, torch.randn(0, self.tokens, self.dim))
        with self.assertRaises(ValueError):
            router(torch.randn(self.tokens, self.dim + 1), self.candidates)
        with self.assertRaises(ValueError):
            router(self.prefix, self.candidates, torch.ones(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
