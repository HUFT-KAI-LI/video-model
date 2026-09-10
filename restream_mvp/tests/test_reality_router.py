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
        self.tokens, self.dim = 8, 12
        # Opposite pooled directions guarantee that the two prefixes switch IDs.
        first = torch.randn(self.tokens, self.dim)
        self.candidates = torch.stack([first, -first, torch.randn_like(first)])
        self.router = FrozenStateRouter()

    def test_top1_preserves_selected_tokens_exactly(self):
        self.assertEqual(list(self.router.parameters()), [])
        for index in (0, 1):
            memory, mask, stats = self.router(self.candidates[index], self.candidates)
            self.assertEqual(stats['selected_indices'].tolist(), [[index]])
            self.assertEqual(memory.shape, (1, 1, 8, 12))
            self.assertTrue(mask.all())
            torch.testing.assert_close(memory[0, 0], self.candidates[index], rtol=0, atol=0)
            self.assertFalse(memory.requires_grad)
        # Temperature only affects diagnostic probabilities, not memory scaling.
        cold = FrozenStateRouter(temperature=.001)(self.candidates[0], self.candidates)
        hot = FrozenStateRouter(temperature=10)(self.candidates[0], self.candidates)
        self.assertFalse(torch.allclose(cold[2]['router_weights'], hot[2]['router_weights']))
        torch.testing.assert_close(cold[0], hot[0], rtol=0, atol=0)

    def test_masked_nan_padding_and_empty_rows_are_inert(self):
        candidates = self.candidates.repeat(2, 1, 1, 1)
        candidates[0, 0] = float('nan')
        candidates[1] = float('nan')
        masks = torch.tensor([[False, True, True], [False, False, False]])
        memory, mask, stats = self.router(self.candidates[0].repeat(2, 1, 1), candidates, masks)
        self.assertTrue(torch.isfinite(memory).all())
        self.assertTrue(torch.isfinite(stats['router_weights']).all())
        self.assertNotEqual(stats['selected_indices'][0, 0], 0)
        self.assertEqual(stats['selected_indices'][1, 0], -1)
        self.assertEqual(mask.tolist(), [[True], [False]])
        self.assertTrue(torch.equal(memory[1], torch.zeros_like(memory[1])))
        model = RealityMemory(self.dim, 8, 16, 2)
        torch.nn.init.normal_(model.output.weight, std=.1)
        context = torch.randn(2, 3, 16)
        fused, _ = model(context, memory, mask)
        torch.testing.assert_close(fused[1], context[1], rtol=0, atol=0)

    def test_selected_reference_survives_projector_and_changes_context(self):
        model = RealityMemory(self.dim, 8, 16, 2)
        torch.nn.init.normal_(model.output.weight, std=.1)
        context = torch.randn(1, 3, 16)
        routed = [self.router(self.candidates[i], self.candidates) for i in (0, 1)]
        fused = [model(context, *r[:2])[0] for r in routed]
        for index in (0, 1):
            direct, _ = model(context, self.candidates[index][None, None], torch.ones(1, 1, dtype=torch.bool))
            torch.testing.assert_close(fused[index], direct, rtol=0, atol=0)
        self.assertFalse(torch.allclose(fused[0], fused[1]))
        self.assertFalse(torch.allclose(model.projector(routed[0][0]), model.projector(routed[1][0])))
        fused[0].square().mean().backward()
        self.assertGreater(model.projector.net[1].weight.grad.norm().item(), 0)

    def test_fresh_zero_init_has_reference_specific_output_gradients(self):
        model = RealityMemory(self.dim, 8, 16, 2)
        context = torch.randn(1, 3, 16)
        gradients = []
        for index in (0, 1):
            memory, mask, _ = self.router(self.candidates[index], self.candidates)
            fused, _ = model(context, memory, mask)
            torch.testing.assert_close(fused, context, rtol=0, atol=0)
            gradients.append(torch.autograd.grad(fused.square().mean(), model.output.weight)[0])
        self.assertGreater(gradients[0].norm().item(), 0)
        self.assertFalse(torch.allclose(*gradients))

    def test_invalid_router_inputs_raise(self):
        for kwargs in ({'mode': 'soft'}, {'mode': 'topk'}, {'temperature': 0}, {'temperature': float('nan')}):
            with self.assertRaises(ValueError):
                FrozenStateRouter(**kwargs)
        for candidates, mask in ((torch.randn(0, self.tokens, self.dim), None),
                                 (self.candidates, torch.ones(2, dtype=torch.bool)),
                                 (torch.full_like(self.candidates, float('nan')), None)):
            with self.assertRaises(ValueError):
                self.router(self.candidates[0], candidates, mask)


if __name__ == '__main__':
    unittest.main()
