import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.wan_said.io_utils import write_json
from experiments.wan_said.pilot_v2.aggregate import official_vbench, save_summary, vbench, videophy2
from experiments.wan_said.pilot_v2.benchmarks import make_jobs, suites
from experiments.wan_said.runtime import ROOT


class VideoPhyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.jobs = make_jobs('videophy2', [dict(id=str(i), number=i, prompt=f'Caption {i}') for i in range(1, 5)])
        self.sa, self.pc = self.root/'sa.csv', self.root/'pc.csv'
        self.rows = [dict(videopath=str(self.root/j['relative_path']), caption=j['original_prompt'], score=s)
                     for j, s in zip(self.jobs, [4, 5, 3, 1])]
        self.write(self.sa, self.rows)
        self.write(self.pc, [dict(r, score=s) for r, s in zip(self.rows, [5, 3, 4, 2])][::-1])

    def write(self, path, rows):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['videopath', 'caption', 'score'])
            writer.writeheader()
            writer.writerows(rows)

    def test_joint_joins_video_identity_not_csv_position(self):
        result = videophy2(self.jobs, self.root, self.sa, self.pc)
        self.assertTrue(result['complete'])
        self.assertEqual(result['scores'], dict(sa_mean=3.25, pc_mean=3.5, sa_at_4=.5, pc_at_4=.5, joint_at_4=.25))
        self.assertEqual(result['denominator'], 4)

    def test_missing_duplicate_unexpected_or_caption_mismatch_withhold_scores(self):
        for rows in (self.rows[:-1], self.rows+[self.rows[0]],
                     [dict(self.rows[0], videopath='/wrong/video.mp4')]+self.rows[1:],
                     [dict(self.rows[0], caption='Changed')]+self.rows[1:]):
            self.write(self.sa, rows)
            result = videophy2(self.jobs, self.root, self.sa, self.pc)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['scores'])
            self.assertEqual(result['denominator'], 4)

    def test_zero_noninteger_nan_infinite_out_of_range_are_failures(self):
        for value in ('', 'nan', 'inf', '-inf', '0', '6', '4.5', 'four'):
            self.write(self.sa, [dict(self.rows[0], score=value)]+self.rows[1:])
            result = videophy2(self.jobs, self.root, self.sa, self.pc)
            self.assertFalse(result['complete'])
            self.assertEqual(result['errors'][0]['reason'], 'invalid_rating')

    def test_failed_summary_removes_stale_comparison_csv(self):
        result = videophy2(self.jobs, self.root, self.sa, self.pc)
        save_summary(result, self.root, 'C')
        self.assertTrue((self.root/'summary.csv').exists())
        self.write(self.sa, self.rows[:-1])
        with self.assertRaises(ValueError):
            save_summary(videophy2(self.jobs, self.root, self.sa, self.pc), self.root, 'C')
        self.assertFalse((self.root/'summary.csv').exists())
        self.assertFalse(json.loads((self.root/'summary.json').read_text())['main_table_ready'])


@unittest.skipUnless((ROOT/'vendor/VBench/scripts/cal_final_score.py').exists(), 'Pinned official VBench scripts required')
class VBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompts = suites()[0]['vbench']
        cls.official, _ = official_vbench()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.jobs = make_jobs('vbench', self.prompts)
        self.result = {}
        for key in self.official['TASK_INFO']:
            dimension = key.replace(' ', '_')
            limits = self.official['NORMALIZE_DIC'][key]
            score = limits['Min']+.7*(limits['Max']-limits['Min'])
            paths = [str(self.root/j['relative_path']) for p in self.prompts if dimension in p['dimensions']
                     for j in make_jobs('vbench', [p])]
            self.result[dimension] = [score, [dict(video_path=p, video_results=.2) for p in paths]]
        self.path = self.root/'test_eval_results.json'

    def aggregate(self, payload=None, paths=None):
        write_json(self.path, self.result if payload is None else payload)
        return vbench(self.jobs, self.root, paths or [self.path])

    def test_uses_upstream_aggregate_not_video_mean_and_exact_official_functions(self):
        result = self.aggregate()
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['raw_dimensions']), 16)
        normalized = self.official['get_nomalized_score']({k.replace('_', ' '): v[0] for k, v in self.result.items()})
        q, s = self.official['get_quality_score'](normalized), self.official['get_semantic_score'](normalized)
        self.assertEqual(result['scores'], dict(quality_score=q, semantic_score=s,
                                               total_score=self.official['get_final_score'](q, s)))
        self.assertAlmostEqual(result['scores']['total_score'], .7)
        self.assertEqual(result['aggregation']['quality_weight'], 4)
        self.assertEqual(result['aggregation']['dimension_weights']['dynamic degree'], .5)
        self.assertEqual(result['coverage']['human_action']['expected'], 500)
        self.assertEqual(result['coverage']['dynamic_degree']['expected'], 360)
        self.assertNotAlmostEqual(result['scores']['total_score'], sum(r[0] for r in self.result.values())/16)

    def test_each_missing_or_extra_dimension_invalidates_total(self):
        missing = dict(self.result)
        missing.pop('scene')
        for payload in (missing, dict(self.result, unknown=[.5, []])):
            result = self.aggregate(payload)
            self.assertFalse(result['complete'])
            self.assertIsNone(result['scores'])

    def test_missing_duplicate_unexpected_per_video_rows_invalidates_total(self):
        dimension = 'background_consistency'
        for change in ('missing', 'duplicate', 'path', 'nan'):
            with self.subTest(change=change):
                payload = json.loads(json.dumps(self.result))
                rows = payload[dimension][1]
                if change == 'missing':
                    rows.pop()
                elif change == 'duplicate':
                    rows.append(rows[0])
                elif change == 'path':
                    rows[0]['video_path'] = '/other/'+Path(rows[0]['video_path']).name
                else:
                    rows[0]['video_results'] = math.nan
                result = self.aggregate(payload)
                self.assertFalse(result['complete'])
                self.assertIsNone(result['scores'])
                self.assertEqual(result['denominator'], 4720)

    def test_boolean_ratings_are_valid_and_aggregates_must_be_finite(self):
        self.result['dynamic_degree'][1][0]['video_results'] = True
        self.assertTrue(self.aggregate()['complete'])
        self.result['dynamic_degree'][0] = math.inf
        self.assertFalse(self.aggregate()['complete'])

    def test_duplicate_result_files_are_not_silently_overwritten(self):
        result = self.aggregate(paths=[self.path, self.path])
        self.assertFalse(result['complete'])
        self.assertTrue(result['errors'])

    def test_color_exclusions_are_visible_not_zero_imputed(self):
        self.result['color'][1].pop()
        result = self.aggregate()
        self.assertFalse(result['complete'])
        self.assertIn('upstream_caveat', result['coverage']['color'])
        self.assertEqual(len(result['coverage']['color']['missing_or_invalid']), 1)

    def test_official_normalization_is_not_clipped(self):
        self.result['scene'][0] = 1.
        result = self.aggregate()
        self.assertGreater(result['normalized_weighted_dimensions']['scene'], 1.)


if __name__ == '__main__':
    unittest.main()
