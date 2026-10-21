"""Synthetic orchestration checks; no student/encoder training is performed here."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.wan_said.io_utils import write_json
from experiments.wan_said.pilot_v2.config import load_config
from experiments.wan_said.pilot_v2 import run_c
from experiments.wan_said.runtime import ROOT


@unittest.skipUnless((ROOT/'datasets/pilot_v2/manifests/split_lock.json').exists(), 'Frozen local pilot required')
class QueueTests(unittest.TestCase):
    def run_fixture(self, root, validated):
        (root/'grounding').mkdir()
        write_json(root/'grounding/report.json', dict(steps=1000))
        torch.save(dict(steps=1000, grounding_validated=validated), root/'grounding/grounding.pt')
        return run_c.run(load_config(), root, 'fixture.service', exploratory=True, benchmarks=())

    def test_weak_grounding_blocks_c_without_running_any_training_child(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(run_c.subprocess, 'check_output', return_value='ActiveState=inactive\nResult=success\nMainPID=0\n'), \
                patch.object(run_c.subprocess, 'run') as execute:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Grounding did not complete'):
                self.run_fixture(root, False)
            execute.assert_not_called()
            status = json.loads((root/'status.json').read_text())
            self.assertEqual(status['stage'], 'failed')
            self.assertFalse(status['complete'])

    def test_validated_grounding_queues_full_train_then_full_val_not_smoke(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(run_c.subprocess, 'check_output', return_value='ActiveState=inactive\nResult=success\nMainPID=0\n'), \
                patch.object(run_c.subprocess, 'run') as execute:
            result = self.run_fixture(Path(tmp), True)
            commands = [call.args[0] for call in execute.call_args_list]
            self.assertEqual([c[2] for c in commands], [run_c.MODULE+'train', run_c.MODULE+'validate', run_c.MODULE+'benchmarks'])
            self.assertIn('--exploratory', commands[0])
            self.assertEqual(commands[0][commands[0].index('--steps')+1], '3995')
            self.assertTrue(all('--smoke' not in c and '--limit' not in c for c in commands))
            self.assertTrue(result['complete'])
            self.assertFalse(result['publication_eligible'])
            self.assertFalse(result['official_scorers_executed'])


if __name__ == '__main__':
    unittest.main()
