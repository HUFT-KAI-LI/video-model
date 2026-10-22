import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.wan_said.io_utils import write_json, write_jsonl
from experiments.wan_said.pilot_v2 import evaluate_c


class OfficialQueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = argparse.Namespace(plans=str(self.root/'plans'), output=str(self.root/'output'),
                    devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], config=None,
                    checkpoint='fixture.pt', minimum_free_gib=50,
                    videophy2_environment=str(self.root/'environment.json'))
        write_jsonl(self.root/'plans/videophy2.jsonl', [dict(job_id=str(i)) for i in range(590)])
        write_json(self.root/'environment.json', dict(weights=[]))
        self.state = dict(suites=dict(videophy2={}))

    def test_four_workers_receive_same_full_plan_and_distinct_devices(self):
        processes = [Mock() for _ in range(4)]
        for i, process in enumerate(processes):
            process.pid = i
            process.poll.return_value = 0
        records = [dict(seconds=10) for _ in range(590)]
        with patch.object(evaluate_c, 'validate_official_plan'), patch.object(
                evaluate_c, 'recorded_jobs', return_value=records), patch.object(
                evaluate_c.subprocess, 'Popen', side_effect=processes) as launch:
            evaluate_c.generate_suite(self.args, 'videophy2', self.state, lambda: None)
        for i, call in enumerate(launch.call_args_list):
            argv = call.args[0]
            self.assertEqual(argv[argv.index('--jobs')+1], str(self.root/'plans/videophy2.jsonl'))
            self.assertEqual(argv[argv.index('--worker-index')+1], str(i))
            self.assertEqual(argv[argv.index('--device')+1], f'cuda:{i}')
            self.assertNotIn('--limit', argv)
            self.assertNotIn('--smoke', argv)
            self.assertNotIn('--smoke-steps', argv)
        progress = json.loads((self.root/'output/videophy2/videos/progress.json').read_text())
        self.assertTrue(progress['complete'])
        self.assertEqual(progress['denominator'], 590)

    def test_worker_failure_terminates_and_reaps_other_workers(self):
        processes = [Mock() for _ in range(4)]
        for process in processes:
            process.pid = 1
            process.poll.return_value = None
        processes[0].poll.return_value = 1
        with patch.object(evaluate_c, 'validate_official_plan'), patch.object(
                evaluate_c, 'recorded_jobs', return_value=[]), patch.object(
                evaluate_c.subprocess, 'Popen', side_effect=processes):
            with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                evaluate_c.generate_suite(self.args, 'videophy2', self.state, lambda: None)
        for process in processes[1:]:
            process.terminate.assert_called_once()
        for process in processes:
            process.wait.assert_called_once()
        self.assertFalse((self.root/'output/videophy2/videos/progress.json').exists())

    def test_successful_exit_with_missing_videos_does_not_mark_generation_complete(self):
        process = Mock(pid=1)
        process.poll.return_value = 0
        with patch.object(evaluate_c, 'validate_official_plan'), patch.object(
                evaluate_c, 'recorded_jobs', return_value=[]), patch.object(
                evaluate_c.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(ValueError, 'Full generation coverage'):
                evaluate_c.generate_suite(self.args, 'videophy2', self.state, lambda: None)

    def test_scoring_failure_never_sets_complete(self):
        root = self.root/'output/videophy2'
        write_json(root/'scores/score_status.json', dict(main_table_ready=False))
        with patch.object(evaluate_c.subprocess, 'run') as invoke:
            with self.assertRaisesRegex(RuntimeError, 'complete comparable summary'):
                evaluate_c.score_suite(self.args, 'videophy2', self.state, lambda: None)
        self.assertNotIn('--aggregate-only', invoke.call_args.args[0])
        self.assertFalse(self.state['suites']['videophy2'].get('scoring_complete'))

    def test_completed_scores_are_revalidated_not_blindly_skipped(self):
        root = self.root/'output/videophy2'
        write_json(root/'scores/score_status.json', dict(main_table_ready=True, scores=dict(joint_at_4=.5)))
        with patch.object(evaluate_c.subprocess, 'run') as invoke:
            evaluate_c.score_suite(self.args, 'videophy2', self.state, lambda: None)
        self.assertIn('--aggregate-only', invoke.call_args.args[0])
        self.assertTrue(self.state['suites']['videophy2']['scoring_complete'])


if __name__ == '__main__':
    unittest.main()
