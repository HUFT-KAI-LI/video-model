import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.wan_said.io_utils import sha256, write_json, write_jsonl
from experiments.wan_said.pilot_v2 import subset
from experiments.wan_said.pilot_v2.evaluate_subset import reuse


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.jobs = [dict(suite='videophy2', job_id=str(i),
                          relative_path=f'videophy2/{i:04d}.mp4') for i in range(590)]
        self.plan = self.root/'full.jsonl'
        self.lock = self.root/'subset.json'
        write_jsonl(self.plan, self.jobs)

    def prepare(self):
        with patch.object(subset, 'validate_official_plan'):
            return subset.prepare(self.plan, self.lock)

    def test_reproducible_exactly_100_distinct_samples_with_full_denominator(self):
        first = self.prepare()
        self.assertEqual(first, self.prepare())
        selected = subset.select(self.jobs, self.plan, self.lock)
        self.assertEqual(len(selected), 100)
        self.assertEqual(len({j['job_id'] for j in selected}), 100)
        self.assertNotEqual(selected, self.jobs[:100])
        self.assertEqual(first['full_benchmark_denominator'], 590)
        self.assertFalse(first['formal_benchmark_complete'])
        self.assertFalse(first['scores_used_for_selection'])

    def test_edited_selection_or_parent_plan_is_rejected(self):
        original = self.prepare()
        write_json(self.lock, dict(original, selected_job_ids=original['selected_job_ids'][::-1]))
        with self.assertRaises(ValueError):
            subset.select(self.jobs, self.plan, self.lock)
        write_json(self.lock, original)
        write_jsonl(self.plan, self.jobs[::-1])
        with self.assertRaises(ValueError):
            subset.select(self.jobs, self.plan, self.lock)

    def test_invalid_sizes_and_non_videophy_subsets_rejected(self):
        for count in (0, 590, 591):
            with self.assertRaises(ValueError):
                subset.selection(self.jobs, 'fixture', count=count)
        with self.assertRaises(ValueError):
            subset.selection([dict(j, suite='vbench') for j in self.jobs], 'fixture')

    def test_reuse_checks_identity_and_hardlinks_only_selected_videos(self):
        source, videos = self.root/'source', self.root/'videos'
        checkpoint = self.root/'checkpoint.pt'
        checkpoint.touch()
        job = self.jobs[0]
        path = source/job['relative_path']
        path.parent.mkdir(parents=True)
        path.touch()
        write_json(source/'generation_lock.json', dict(group='C', smoke=False,
                   checkpoint_sha256=sha256(checkpoint), jobs_sha256=sha256(self.plan), protocol={}))
        write_json(source/'records/0.json', dict(job=job, sha256=sha256(path)))
        with patch('experiments.wan_said.pilot_v2.evaluate_subset.verify_video'), patch(
                'experiments.wan_said.pilot_v2.evaluate_subset.native_generation_protocol', return_value={}):
            self.assertEqual(reuse(self.jobs[:2], source, videos, checkpoint, self.plan), ['0'])
            self.assertEqual(os.stat(path).st_ino, os.stat(videos/job['relative_path']).st_ino)
            record = json.loads((videos/'records/0.json').read_text())
            self.assertEqual(record['reused_from'], str(path.resolve()))
            checkpoint.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                reuse(self.jobs[:2], source, videos, checkpoint, self.plan)


if __name__ == '__main__':
    unittest.main()
