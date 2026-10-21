"""Exercise validation plumbing with tiny synthetic encoders/DiT, not real-model scores."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.wan_said.alignment import Grounding, Projector
from experiments.wan_said.io_utils import sha256
from experiments.wan_said.pilot_v2.config import load_config
from experiments.wan_said.pilot_v2 import validate as validation


class TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(15)])
        self.factor = 1.

    def forward(self, latents):
        hidden = latents.mean().reshape(1, 1, 1).expand(1, 2, 1536)
        for block in self.blocks:
            hidden = block(hidden)
        return latents*.1+self.factor*hidden.mean()


class ValidationWorkflowTests(unittest.TestCase):
    def test_all_val_rows_paired_resumable_and_test_never_read(self):
        cfg = load_config()
        cfg['devices'] = {k: 'cpu' for k in cfg['devices']}
        identity = {'fixture': True}
        rows = [dict(sample_id=f'val_{i}', split='val') for i in range(3)]
        calls, splits, noisy_inputs = [], [], []

        class Provider:
            def __init__(self, cfg):
                pass

            def __call__(self, row, kinds):
                calls.append((row['sample_id'], kinds))
                index = int(row['sample_id'][-1])+1
                features = dict(text=dict(pooled=torch.ones(1, 4096)*index,
                                          context=torch.ones(1, 2, 4096)*index),
                                teacher=dict(pooled=torch.ones(1, 768)*index))
                if 'vae' in kinds:
                    features['vae'] = dict(latents=torch.ones(1, 16, 2, 4, 4)*index)
                return features

        def load_split(manifests, split):
            splits.append(split)
            self.assertEqual(split, 'val')
            return rows

        def load_adapter(model, state):
            model.factor = state['fixture_factor']

        def forward(dit, latents, **kwargs):
            noisy_inputs.append((latents.clone(), kwargs['timestep'].clone()))
            return dit(latents)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            grounding, checkpoint = root/'ground.pt', root/'c.pt'
            torch.save(dict(model=Grounding(video_dim=768, dim=cfg['dim']).state_dict(),
                            contract=dict(identity=identity), mode='said', grounding_validated=True), grounding)
            torch.save(dict(group='C', contract=dict(identity=identity, grounding_sha256=sha256(grounding),
                            smoke=False, exploratory=True), lora=dict(fixture_factor=2.),
                            projector=Projector(1536, cfg['dim']).state_dict()), checkpoint)
            with patch.object(validation, 'load_split', load_split), \
                    patch.object(validation, 'experiment_identity', return_value=identity), \
                    patch.object(validation, 'load_dit', side_effect=lambda *a, **k: TinyDiT()), \
                    patch.object(validation, 'load_trainable', load_adapter), \
                    patch.object(validation, 'forward_fn', return_value=forward), \
                    patch.object(validation, 'OnlineFeatures', Provider):
                result = validation.validate(cfg, checkpoint, grounding, root/'validation')
                self.assertTrue(result['complete'])
                self.assertEqual(result['denominator'], 3)
                self.assertFalse(result['test_used'])
                self.assertFalse(result['official_scorers_executed'])
                self.assertFalse(result['publication_eligible'])
                before = json.loads((root/'validation/summary.json').read_text())['scores']
                resumed = validation.validate(cfg, checkpoint, grounding, root/'validation')
                self.assertEqual(before, resumed['scores'])
                self.assertEqual(splits, ['val', 'val'])
                self.assertEqual(len(calls), 6)
                self.assertTrue(all('vae' in kinds for _, kinds in calls[:3]))
                self.assertTrue(all('vae' not in kinds for _, kinds in calls[3:]))
                self.assertEqual(len(noisy_inputs), 6)
                for index in range(0, 6, 2):
                    self.assertTrue(torch.equal(noisy_inputs[index][0], noisy_inputs[index+1][0]))
                    self.assertTrue(torch.equal(noisy_inputs[index][1], noisy_inputs[index+1][1]))


if __name__ == '__main__':
    unittest.main()
