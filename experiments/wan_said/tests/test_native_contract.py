import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from experiments.wan_said.alignment import HiddenCapture, Projector, distillation_loss
from experiments.wan_said.runtime import ROOT, forward_fn, upstream


@unittest.skipUnless((ROOT / 'vendor/DiffSynth-Studio').is_dir(), 'Pinned vendor checkout required')
class NativeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        upstream()

    def test_forward_is_unmodified_native_function(self):
        from diffsynth.pipelines.wan_video import model_fn_wan_video

        self.assertIs(forward_fn(), model_fn_wan_video)

    def test_h15_capture_preserves_native_fm_and_gradients(self):
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        from diffsynth.diffusion.loss import FlowMatchSFTLoss

        class TinyStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(30)])

            def forward(self, x, use_checkpoint):
                for block in self.blocks:
                    update = checkpoint(block, x, use_reentrant=False) if use_checkpoint else block(x)
                    x = x + .01 * update.tanh()
                return x

        for use_checkpoint in (False, True):
            with self.subTest(checkpoint=use_checkpoint):
                torch.manual_seed(42)
                model = TinyStudent()
                head = Projector(4, 8)
                latent = torch.randn(1, 3, 4)
                target = torch.randn(1, 8)
                scheduler = FlowMatchScheduler('Wan')
                scheduler.set_timesteps(1000, training=True)
                observed = []

                def native_forward(dit, latents, timestep, **inputs):
                    prediction = dit(latents, use_checkpoint)
                    observed.append((latents.detach().clone(), timestep.detach().clone(),
                                     prediction.detach().clone()))
                    return prediction

                pipe = SimpleNamespace(scheduler=scheduler, torch_dtype=torch.float32,
                                       device='cpu', dit=model, in_iteration_models=('dit',),
                                       model_fn=native_forward)
                torch.manual_seed(123)
                reference = FlowMatchSFTLoss(pipe, input_latents=latent)
                reference.backward()
                reference_grads = {name: p.grad.clone() for name, p in model.named_parameters()}
                model.zero_grad(set_to_none=True)

                torch.manual_seed(123)
                with HiddenCapture(model.blocks[14]) as capture:
                    actual = FlowMatchSFTLoss(pipe, input_latents=latent)
                auxiliary = distillation_loss(head(capture.hidden), target)
                self.assertTrue(torch.isfinite(auxiliary))
                actual.backward(retain_graph=True)
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                for before, after in zip(observed[0], observed[1]):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in head.parameters()))

                model.zero_grad(set_to_none=True)
                auxiliary.backward()
                self.assertTrue(all(p.grad is not None for b in model.blocks[:15] for p in b.parameters()))
                self.assertTrue(all(p.grad is None for b in model.blocks[15:] for p in b.parameters()))
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
                self.assertEqual(len(model.blocks[14]._forward_hooks), 0)


if __name__ == '__main__':
    unittest.main()
