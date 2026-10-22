import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.wan_said.io_utils import sha256, write_json
from experiments.wan_said.pilot_v2 import scorer_probe


class ScorerProbeTests(unittest.TestCase):
    def exercise(self, rating):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        video = root/'video.mp4'
        video.touch()
        write_json(root/'record.json', dict(path=str(video), sha256=sha256(video),
                                          job=dict(original_prompt='Caption')))
        write_json(root/'environment.json', dict(weights=[dict(path=str(video), sha256=sha256(video))]))
        output = root/'probe'

        def run(*args, **kwargs):
            for mode in ('sa', 'pc'):
                with (output/f'videophy2_{mode}.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=['videopath', 'caption', 'score'])
                    writer.writeheader()
                    writer.writerow(dict(videopath=str(video), caption='Caption', score=rating))
            return Mock(returncode=0)

        mocks = [patch.object(scorer_probe, 'verify_video'),
                 patch.object(scorer_probe, 'native_generation_protocol', return_value={}),
                 patch.object(scorer_probe, 'source_hash'), patch.object(scorer_probe, 'check_vendor'),
                 patch.object(scorer_probe, 'recipes', return_value=[dict(script=str(video), argv=['fixture'],
                              cwd=str(root), name='fixture')]),
                 patch.object(scorer_probe.subprocess, 'run', side_effect=run)]
        for mock in mocks:
            mock.start()
            self.addCleanup(mock.stop)
        return root, output

    def test_success_is_only_integration_evidence_not_a_benchmark_score(self):
        root, output = self.exercise(4)
        scorer_probe.probe(root/'record.json', root/'environment.json', output)
        status = json.loads((output/'probe_status.json').read_text())
        self.assertTrue(status['integration_succeeded'])
        self.assertFalse(status['formal_benchmark_complete'])
        self.assertFalse(status['scores_are_benchmark_results'])
        self.assertEqual(status['benchmark_denominator'], 590)
        self.assertFalse((output/'summary.csv').exists())

    def test_zero_parse_failure_does_not_pass_integration(self):
        root, output = self.exercise(0)
        with self.assertRaisesRegex(ValueError, 'unparseable'):
            scorer_probe.probe(root/'record.json', root/'environment.json', output)
        self.assertFalse(json.loads((output/'probe_status.json').read_text()).get('integration_succeeded'))


if __name__ == '__main__':
    unittest.main()
