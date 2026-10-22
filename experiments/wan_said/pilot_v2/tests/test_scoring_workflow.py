"""Synthetic transport fixtures, never evidence of real AutoEval model execution."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.wan_said.io_utils import sha256, write_json, write_jsonl
from experiments.wan_said.pilot_v2.benchmarks import make_jobs
from experiments.wan_said.pilot_v2 import score, subset


class ScoringWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output = self.root/'scores'
        self.videos = self.root/'videos'
        self.videos.mkdir()
        self.jobs = make_jobs('videophy2', [dict(id='one', number=1, prompt='First'),
                                           dict(id='two', number=2, prompt='Second')])
        self.jobs_path = self.root/'jobs.jsonl'
        write_jsonl(self.jobs_path, self.jobs)
        write_json(self.videos/'generation_lock.json', dict(smoke=False, group='C', exploratory=True,
                   jobs_sha256=sha256(self.jobs_path), protocol={'fixture': True}))
        (self.videos/'records').mkdir()
        for job in self.jobs:
            path = self.videos/job['relative_path']
            path.parent.mkdir(exist_ok=True)
            path.touch()
            write_json(self.videos/'records'/(job['job_id']+'.json'), dict(job=job, sha256=sha256(path)))
        self.env = self.root/'env.json'
        write_json(self.env, dict(weights=[dict(path=str(self.jobs_path), sha256=sha256(self.jobs_path))]))
        self.tasks = [dict(name=mode, script=str(self.jobs_path), script_sha256=sha256(self.jobs_path),
                           cwd=str(self.root), argv=['/bin/true']) for mode in ('sa', 'pc')]
        for target, value in [('recipes', lambda *args: [dict(t) for t in self.tasks]),
                              ('validate_official_plan', lambda *args: None),
                              ('native_generation_protocol', lambda: {'fixture': True}),
                              ('verify_video', lambda *args: None),
                              ('check_vendor', lambda *args: self.root),
                              ('source_hash', lambda *args: 'fixture')]:
            mock = patch.object(score, target, value)
            mock.start()
            self.addCleanup(mock.stop)

    def write_results(self, missing=False):
        self.output.mkdir(exist_ok=True)
        for mode in ('sa', 'pc'):
            with (self.output/f'videophy2_{mode}.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['videopath', 'caption', 'score'])
                writer.writeheader()
                jobs = self.jobs[:1] if missing and mode == 'pc' else self.jobs[::-1]
                writer.writerows(dict(videopath=str(self.videos/j['relative_path']),
                                      caption=j['original_prompt'], score=4) for j in jobs)

    def execute(self, **extra):
        return score.run('videophy2', self.jobs_path, self.videos, self.output, self.env, **extra)

    def test_scoring_automatically_aggregates_and_can_reaggregate(self):
        self.write_results()
        status = self.execute()
        self.assertTrue(status['main_table_ready'])
        self.assertEqual(status['scores']['joint_at_4'], 1.)
        result = json.loads((self.output/'summary.json').read_text())
        self.assertTrue(result['exploratory'])
        self.assertFalse(result['publication_eligible'])
        self.assertTrue((self.output/'summary.csv').is_file())
        status = self.execute(aggregate_only=True)
        self.assertTrue(status['main_table_ready'])

    def test_incomplete_results_preserve_audit_and_withhold_total(self):
        self.write_results(missing=True)
        with self.assertRaises(ValueError):
            self.execute()
        status = json.loads((self.output/'score_status.json').read_text())
        self.assertFalse(status['main_table_ready'])
        self.assertTrue(status['official_scorers_executed'])
        self.assertIsNone(json.loads((self.output/'summary.json').read_text())['scores'])
        self.assertFalse((self.output/'summary.csv').exists())

    def test_failed_official_task_does_not_aggregate(self):
        self.tasks[0]['argv'] = ['/bin/false']
        self.write_results()
        with self.assertRaises(RuntimeError):
            self.execute()
        self.assertFalse((self.output/'summary.json').exists())
        status = json.loads((self.output/'score_status.json').read_text())
        self.assertFalse(status['official_scorers_executed'])

    def test_aggregate_only_rejects_changed_generation_lock(self):
        self.write_results()
        self.execute()
        generation = json.loads((self.videos/'generation_lock.json').read_text())
        write_json(self.videos/'generation_lock.json', dict(generation, group='A'))
        with self.assertRaisesRegex(ValueError, 'generation_lock_sha256'):
            self.execute(aggregate_only=True)
        self.assertFalse((self.output/'summary.csv').exists())
        self.assertFalse(json.loads((self.output/'score_status.json').read_text())['main_table_ready'])

    def test_subset_aggregates_without_claiming_full_benchmark_or_allowing_relabelling(self):
        lock = self.root/'subset.json'
        write_json(lock, subset.selection(self.jobs, sha256(self.jobs_path), count=1, seed=42))
        self.jobs = subset.select(self.jobs, self.jobs_path, lock)
        generation = json.loads((self.videos/'generation_lock.json').read_text())
        write_json(self.videos/'generation_lock.json', dict(generation, subset_lock_sha256=sha256(lock)))
        self.write_results()
        status = self.execute(subset_lock=lock)
        self.assertTrue(status['official_scorers_executed'])
        self.assertFalse(status['main_table_ready'])
        self.assertTrue(status['subset_table_ready'])
        self.assertFalse(status['formal_benchmark_complete'])
        report = json.loads((self.output/'summary.json').read_text())
        self.assertEqual(report['denominator'], 1)
        self.assertEqual(report['full_benchmark_denominator'], 2)
        self.assertFalse(report['publication_eligible'])
        with (self.output/'summary.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['evaluation_scope'], 'subset_pilot')
        self.assertEqual(row['formal_benchmark_complete'], 'False')
        self.assertTrue(self.execute(subset_lock=lock, aggregate_only=True)['subset_table_ready'])
        with self.assertRaises(ValueError):
            self.execute()

    def test_aggregate_only_rejects_edited_raw_scores_and_clears_old_table(self):
        self.write_results()
        self.execute()
        self.write_results(missing=True)
        with self.assertRaisesRegex(ValueError, 'raw scoring results changed'):
            self.execute(aggregate_only=True)
        self.assertFalse((self.output/'summary.csv').exists())


if __name__ == '__main__':
    unittest.main()
