import json
from multiprocessing import get_context
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.wan_said.io_utils import sha256, write_json
from experiments.wan_said.pilot_v2.inference import lock_generation, worker_jobs
from experiments.wan_said.pilot_v2.evaluate_c import recorded_jobs


def acquire(root):
    lock_generation(Path(root), {'fixture': 'one shared protocol'})


class GenerationWorkerTests(unittest.TestCase):
    def test_full_590_and_4720_have_disjoint_complete_four_way_coverage(self):
        for count in (100, 590, 4720):
            jobs = list(range(count))
            parts = [worker_jobs(jobs, rank, 4) for rank in range(4)]
            self.assertEqual(sorted(sum(parts, [])), jobs)
            self.assertEqual(len(set(sum(parts, []))), count)
            self.assertLessEqual(max(map(len, parts))-min(map(len, parts)), 1)

    def test_invalid_worker_rejected(self):
        for index, count in ((0, 0), (-1, 4), (4, 4)):
            with self.assertRaises(ValueError):
                worker_jobs([1], index, count)

    def test_four_process_contract_creation_is_race_free_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as root:
            context = get_context('fork')
            processes = [context.Process(target=acquire, args=(root,)) for _ in range(4)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(json.loads((Path(root)/'generation_lock.json').read_text()),
                             {'fixture': 'one shared protocol'})
            with self.assertRaises(ValueError):
                lock_generation(Path(root), {'fixture': 'changed'})

    def test_only_matching_recorded_videos_count_and_checksums_are_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = dict(job_id='one', relative_path='one.mp4')
            (root/'one.mp4').touch()
            self.assertEqual(recorded_jobs([job], root), [])
            write_json(root/'records/one.json', dict(job=job, sha256=sha256(root/'one.mp4')))
            with patch('experiments.wan_said.pilot_v2.evaluate_c.verify_video'), patch(
                    'experiments.wan_said.pilot_v2.evaluate_c.native_generation_protocol', return_value={}):
                self.assertEqual(len(recorded_jobs([job], root, verify=True)), 1)
                (root/'one.mp4').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    recorded_jobs([job], root, verify=True)


if __name__ == '__main__':
    unittest.main()
