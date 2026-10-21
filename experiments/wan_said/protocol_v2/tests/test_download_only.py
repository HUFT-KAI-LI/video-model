import csv
import io
import json
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np

from experiments.wan_said.io_utils import write_jsonl
from experiments.wan_said.protocol_v2.official_data import (HERE,download,official_rows,
                                                           verify_file)


class OfficialDataTests(unittest.TestCase):
    def test_official_counts_and_non_subset_are_measured(self):
        a = official_rows(HERE/'official/openvid_3w2.csv.gz')
        b = official_rows(HERE/'official/openvid_6w4.csv.gz')
        self.assertEqual((len(a),len(b)),(32000,64000))
        self.assertFalse({r['video'] for r in a} <= {r['video'] for r in b})

    def test_csv_rejects_bad_schema_duplicate_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'list.csv'
            for fields,rows in [(['video','caption'],[['a.mp4','text']]),
                                (['video','caption','seconds'],[['../a.mp4','text',3]]),
                                (['video','caption','seconds'],[['a.mp4','text',3]]*2)]:
                with path.open('w',newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(fields)
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    official_rows(path)

    @staticmethod
    def fixture(root):
        path = root/'fixture.mp4'
        with av.open(str(path),'w') as container:
            stream = container.add_stream('libx264',rate=8)
            stream.width,stream.height,stream.pix_fmt = 48,32,'yuv420p'
            for i in range(6):
                frame = av.VideoFrame.from_ndarray(np.full((32,48,3),i*20,dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        content = path.read_bytes()
        row = dict(name='01234567890_0_0to6.mp4',member='videos/01234567890_0_0to6.mp4',
                   archive='OpenVid_part30.zip',bytes=len(content),crc32=zlib.crc32(content),
                   caption='A changing gray scene.',source_id='youtube:01234567890',
                   duration=.75,family='panda',caption_scope='full_source_video',official_32k=False)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as archive:
            archive.writestr(row['member'],content)
        payload = buffer.getvalue()
        lock = dict(raw_bytes=len(content),official_csv_rows={'32k':1,'64k':1},cache_estimate_bytes={})
        write_jsonl(root/'download_plan.jsonl',[row])
        return row,lock,payload

    def test_crc_sha_full_decode_resume_and_corruption_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row,lock,payload = self.fixture(root)
            def archive(*args,**kwargs):
                return zipfile.ZipFile(io.BytesIO(payload))
            with patch('experiments.wan_said.protocol_v2.official_data.plan',return_value=lock):
                with patch('experiments.wan_said.protocol_v2.official_data.RemoteZip',side_effect=archive):
                    first = download(root,workers=1,attempts=1)
                self.assertTrue(first['full_dataset_ready'])
                self.assertEqual(first['total_cache_bytes'],0)
                self.assertFalse(first['training_enabled'])
                with patch('experiments.wan_said.protocol_v2.official_data.RemoteZip',side_effect=AssertionError('No refetch expected')):
                    self.assertTrue(download(root,workers=1,attempts=1)['full_dataset_ready'])
                path = root/'videos'/row['name']
                with path.open('r+b') as file:
                    file.write(b'broken')
                with patch('experiments.wan_said.protocol_v2.official_data.RemoteZip',side_effect=archive):
                    self.assertTrue(download(root,workers=1,attempts=1)['full_dataset_ready'])
                self.assertTrue(list((root/'videos').glob('*.corrupt.*')))
                self.assertEqual(verify_file(row,path)['decoded_frames'],6)

    def test_failed_file_cannot_mark_complete_and_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row,lock,_ = self.fixture(root)
            with patch('experiments.wan_said.protocol_v2.official_data.plan',return_value=lock), \
                 patch('experiments.wan_said.protocol_v2.official_data.RemoteZip',side_effect=OSError('Network unavailable')):
                with self.assertRaises(RuntimeError):
                    download(root,workers=1,attempts=1)
            progress = json.loads((root/'progress.json').read_text())
            self.assertFalse(progress['full_dataset_ready'])
            self.assertEqual((progress['failed_count'],progress['missing_count']),(1,1))
            self.assertEqual(json.loads((root/'errors.json').read_text())[0]['name'],row['name'])

    def test_space_guard_never_starts_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _,lock,_ = self.fixture(root)
            with patch('experiments.wan_said.protocol_v2.official_data.plan',return_value=lock), \
                 patch('experiments.wan_said.protocol_v2.official_data.shutil.disk_usage',return_value=SimpleNamespace(free=0)), \
                 patch('experiments.wan_said.protocol_v2.official_data.RemoteZip') as remote:
                with self.assertRaises(OSError):
                    download(root,workers=1)
                remote.assert_not_called()
            self.assertFalse(json.loads((root/'storage_blocked.json').read_text())['full_dataset_ready'])


if __name__ == '__main__':
    unittest.main()
