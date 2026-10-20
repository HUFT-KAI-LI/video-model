import os
import tempfile
import unittest
from pathlib import Path

from experiments.wan_said.data import load_split
from experiments.wan_said.io_utils import write_json, write_jsonl
from experiments.wan_said.openvid_subset import prepare_manifests


@unittest.skipUnless(os.environ.get('WAN_SAID_REAL_VIDEO_TEST_ROOT'), 'Opt-in real downloaded-video integration test')
class RealSplitTest(unittest.TestCase):
    def test_downloaded_video_qc_and_frozen_source_split(self):
        import json
        source = Path(os.environ['WAN_SAID_REAL_VIDEO_TEST_ROOT']) / 'videos'
        records = [json.loads(p.read_text()) for p in sorted(source.glob('*.json'))[:40]]
        self.assertEqual(len(records), 40)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_jsonl(root / 'download_plan.jsonl', records)
            write_jsonl(root / 'downloaded.jsonl', records)
            write_json(root / 'selection.json', {'purpose': 'Real-video split integration smoke, not pilot data'})
            output = root / 'manifests'
            prepare_manifests(root, output, workers=4)
            splits = {s: load_split(output, s) for s in ('train', 'val', 'test')}
            self.assertTrue(all(splits.values()))
            self.assertGreaterEqual(sum(map(len, splits.values())), 36)
            self.assertEqual(len(splits['train']), round(sum(map(len, splits.values())) * .8))
            self.assertEqual(len(load_split(output, 'train', training=True)), len(splits['train']))
            with self.assertRaises(ValueError):
                load_split(output, 'test', training=True)


if __name__ == '__main__':
    unittest.main()
