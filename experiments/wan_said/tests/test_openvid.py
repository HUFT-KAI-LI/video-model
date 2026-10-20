import unittest
from unittest.mock import MagicMock, patch

from experiments.wan_said.mirror_download import StrictRangeSession, resolve_url
from experiments.wan_said.openvid_subset import source_identity


class OpenVidTests(unittest.TestCase):
    def test_original_youtube_grouping_across_clips_and_sources(self):
        a, family = source_identity('abc_-deFG12_9_0to100.mp4')
        b, _ = source_identity('abc_-deFG12_99_200to300.mp4')
        c, _ = source_identity('celebv_abc_-deFG12_0.mp4')
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(family, 'panda')

    def test_stock_parent_not_segment(self):
        a, _ = source_identity('pixabay_Ancient_Town_Night_175816_016.mp4')
        b, _ = source_identity('pixabay_Ancient_Town_Night_175816_099.mp4')
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            source_identity('unknown_123.mp4')

    def test_range_ignored_refuses_unbounded_archive(self):
        response = MagicMock(status_code=200, headers={'Content-Length': '40000000000'})
        with patch('requests.Session.request', return_value=response):
            with self.assertRaises(RuntimeError):
                StrictRangeSession().get('https://hf-mirror.com/example', headers={'Range': 'bytes=0-99'})
        response.close.assert_called_once()

    def test_range_and_pinned_mirror_url(self):
        response = MagicMock(status_code=206, headers={'Content-Range': 'bytes 0-99/1000'})
        with patch('requests.Session.request', return_value=response):
            actual = StrictRangeSession().get('https://hf-mirror.com/example', headers={'Range': 'bytes=0-99'})
        self.assertIs(actual, response)
        self.assertEqual(resolve_url('nkp37/OpenVid-1M', 'abc123', 'data.csv'),
                         'https://hf-mirror.com/datasets/nkp37/OpenVid-1M/resolve/abc123/data.csv')


if __name__ == '__main__':
    unittest.main()
