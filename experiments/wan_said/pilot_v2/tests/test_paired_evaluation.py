"""Synthetic transport/aggregation checks; these scores are not model results."""
import argparse
from collections import Counter
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from experiments.wan_said.io_utils import sha256, write_json, write_jsonl
from experiments.wan_said.pilot_v2 import score, subset
from experiments.wan_said.pilot_v2.aggregate import official_vbench, vbench
from experiments.wan_said.pilot_v2.benchmarks import make_jobs, suites
from experiments.wan_said.pilot_v2.evaluate_c import generate_suite

comparison = importlib.import_module('experiments.wan_said.pilot_v2.paired_eval.compare')
queue = importlib.import_module('experiments.wan_said.pilot_v2.paired_eval.run')


class VBenchSubsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets, cls.audit = suites()
        cls.full = make_jobs('vbench', cls.sets['vbench'])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan, self.lock = self.root/'vbench.jsonl', self.root/'subset.json'
        write_jsonl(self.plan, self.full)
        self.contract = subset.prepare(self.plan, self.lock)
        self.jobs = subset.select(self.full, self.plan, self.lock)

    def test_exact_100_prompt_groups_five_seeds_and_all_16_dimensions(self):
        self.assertEqual(self.contract, subset.prepare(self.plan, self.lock))
        self.assertEqual(len(self.jobs), 500)
        self.assertEqual(set(Counter(j['prompt_id'] for j in self.jobs).values()), {5})
        self.assertEqual(len(self.contract['selected_prompt_ids']), 100)
        self.assertEqual(len(self.contract['prompt_groups_per_dimension']), 16)
        self.assertGreaterEqual(min(self.contract['prompt_groups_per_dimension'].values()), 5)
        self.assertEqual(self.contract['full_benchmark_denominator'], 4720)
        self.assertFalse(self.contract['formal_benchmark_complete'])

    def test_modified_group_selection_or_missing_seed_is_rejected(self):
        saved = dict(self.contract, selected_job_ids=self.contract['selected_job_ids'][:-1])
        write_json(self.lock, saved)
        with self.assertRaises(ValueError):
            subset.select(self.full, self.plan, self.lock)
        with self.assertRaises(ValueError):
            subset.vbench_selection([j for j in self.full if j['job_id'] != self.jobs[0]['job_id']],
                                    'fixture', count=100)

    def test_metadata_keeps_original_auxiliary_targets_and_16_scoring_tasks(self):
        status = score.run('vbench', self.plan, self.root/'videos', self.root/'scores',
                           plan_only=True, subset_lock=self.lock)
        metadata = json.loads((self.root/'scores/vbench_subset_metadata.json').read_text())
        self.assertEqual(len({r['prompt_en'] for r in metadata}), 100)
        self.assertTrue(any('auxiliary_info' in row for row in metadata))
        self.assertEqual(len(status['tasks']), 16)
        for task in status['tasks']:
            argv = task['argv']
            self.assertEqual(argv[argv.index('--mode')+1], 'vbench_standard')
            self.assertEqual(len(argv[argv.index('--dimension')+1:]), 1)
            self.assertIn('vbench_subset_metadata.json', argv[argv.index('--full_json_dir')+1])

    def fixture_results(self):
        metadata = subset.vbench_metadata(self.jobs)
        dimensions = self.audit['vbench']['dimensions']
        payload = {d: [.5, [dict(video_path=str(self.root/'videos/vbench'/f'{row["prompt_en"]}-{i}.mp4'),
                                     video_results=.5)
                           for row in metadata if d in row['dimension'] for i in range(5)]] for d in dimensions}
        return metadata, payload

    def test_subset_uses_official_weighted_aggregation_without_relaxing_other_coverage(self):
        metadata, payload = self.fixture_results()
        path = self.root/'fixture_eval_results.json'
        write_json(path, payload)
        result = vbench(self.jobs, self.root/'videos', [path], metadata=metadata)
        self.assertTrue(result['complete'])
        self.assertEqual(result['denominator'], 500)
        official, _ = official_vbench()
        normalized = official['get_nomalized_score']({d.replace('_', ' '): .5 for d in payload})
        self.assertEqual(result['scores']['total_score'], official['get_final_score'](
            official['get_quality_score'](normalized), official['get_semantic_score'](normalized)))
        payload['scene'][1].pop()
        write_json(path, payload)
        result = vbench(self.jobs, self.root/'videos', [path], metadata=metadata,
                        allow_official_color_exclusions=True)
        self.assertFalse(result['complete'])
        self.assertIsNone(result['scores'])

    def test_official_color_exclusion_is_explicit_not_zero_imputed_or_coverage_complete(self):
        metadata, payload = self.fixture_results()
        omitted = payload['color'][1].pop()['video_path']
        path = self.root/'fixture_eval_results.json'
        write_json(path, payload)
        self.assertFalse(vbench(self.jobs, self.root/'videos', [path], metadata=metadata)['complete'])
        result = vbench(self.jobs, self.root/'videos', [path], metadata=metadata,
                        allow_official_color_exclusions=True)
        self.assertTrue(result['complete'])
        self.assertFalse(result['strict_coverage_complete'])
        self.assertFalse(result['coverage']['color']['complete'])
        self.assertEqual(result['official_conditional_exclusions']['color'], [omitted])
        self.assertEqual(result['raw_dimensions']['color'], .5)
        self.assertEqual(result['coverage']['color']['expected']-result['coverage']['color']['scored'], 1)


class PairedRuntimeTests(unittest.TestCase):
    def test_base_generators_have_no_adapter_or_exploratory_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(plans=str(root), output=str(root/'out'), group='base',
                checkpoint=None, config=None, minimum_free_gib=50, devices=[f'cuda:{i}' for i in range(4)])
            write_jsonl(root/'videophy2.jsonl', [dict(job_id='fixture')])
            process = Mock(pid=1)
            process.poll.return_value = 0
            with patch('experiments.wan_said.pilot_v2.evaluate_c.validate_official_plan'), patch(
                    'experiments.wan_said.pilot_v2.evaluate_c.recorded_jobs', return_value=[dict(seconds=1)]), patch(
                    'experiments.wan_said.pilot_v2.evaluate_c.subprocess.Popen', return_value=process) as launch:
                generate_suite(args, 'videophy2', dict(suites=dict(videophy2={})), lambda: None)
            for call in launch.call_args_list:
                argv = call.args[0]
                self.assertEqual(argv[argv.index('--group')+1], 'base')
                self.assertNotIn('--checkpoint', argv)
                self.assertNotIn('--exploratory', argv)

    def test_parallel_scorers_receive_distinct_visible_devices(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = [dict(name=f'd{i}', argv=['/bin/true'], cwd=directory) for i in range(16)]
            processes = [Mock() for _ in tasks]
            for process in processes:
                process.poll.return_value = 0
            with patch.object(score.subprocess, 'Popen', side_effect=processes) as launch:
                score.execute_tasks(tasks, Path(directory), {}, ['0', '1', '2', '3'], lambda: None)
            devices = [call.kwargs['env']['CUDA_VISIBLE_DEVICES'] for call in launch.call_args_list]
            self.assertEqual(devices, ['0', '1', '2', '3']*4)
            self.assertTrue(all(t['returncode'] == 0 for t in tasks))

    def test_failed_parallel_scorer_terminates_and_reaps_siblings(self):
        with tempfile.TemporaryDirectory() as directory:
            processes = [Mock() for _ in range(4)]
            processes[0].poll.return_value = 1
            for process in processes[1:]:
                process.poll.return_value = None
            tasks = [dict(name=f'd{i}', argv=['/bin/false'], cwd=directory) for i in range(4)]
            with patch.object(score.subprocess, 'Popen', side_effect=processes):
                with self.assertRaises(RuntimeError):
                    score.execute_tasks(tasks, Path(directory), {}, ['0', '1', '2', '3'], lambda: None)
            for process in processes[1:]:
                process.terminate.assert_called_once()
                process.wait.assert_called_once()

    def test_paired_bootstrap_joins_ids_and_rejects_unmatched_samples(self):
        rows = [dict(job_id=str(i), sa=3, pc=4) for i in range(100)]
        base = dict(paired_results=rows)
        adapted = dict(paired_results=[dict(r, sa=4) for r in rows[::-1]])
        result = comparison.paired_intervals(base, adapted, resamples=100)
        self.assertEqual(result['sa_mean']['delta'], 1)
        self.assertEqual(result['joint_at_4']['ci95'], [1, 1])
        self.assertEqual(result['pc_mean']['tied'], 100)
        adapted['paired_results'].pop()
        with self.assertRaises(ValueError):
            comparison.paired_intervals(base, adapted)


class PairedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plans, self.locks, self.results, self.reports = {}, {}, {}, {}
        for suite in ('videophy2', 'vbench'):
            self.plans[suite] = self.root/f'{suite}.jsonl'
            self.locks[suite] = self.root/f'{suite}_subset.json'
            write_jsonl(self.plans[suite], [dict(fixture=True)])
            write_json(self.locks[suite], dict(fixture=True))
            self.results[suite], self.reports[suite] = {}, {}
            for arm in ('base', 'C'):
                path = self.root/f'{suite}_{arm}.json'
                write_json(path, dict(fixture=True))
                self.results[suite][arm] = path
                report = dict(jobs_sha256='same', subset_lock_sha256='same', weight_lock_sha256='same',
                    generation=dict(base_weights={'fixture': 'same'}, protocol={'seeded': True}),
                    scores=dict(sa_mean=3., pc_mean=4., sa_at_4=0., pc_at_4=1., joint_at_4=0.) if
                        suite == 'videophy2' else dict(quality_score=.5, semantic_score=.5, total_score=.5))
                if suite == 'videophy2':
                    report['paired_results'] = [dict(job_id=str(i), sa=3, pc=4) for i in range(100)]
                else:
                    report.update(subset_metadata_sha256='same', prompt_count=100, samples_per_prompt=5,
                        raw_dimensions={str(i): .5 for i in range(16)},
                        normalized_weighted_dimensions={str(i): .5 for i in range(16)},
                        coverage={str(i): dict(expected=5, scored=5) for i in range(16)})
                self.reports[suite][arm] = report

    def execute(self):
        with patch.object(comparison, 'select', return_value=[]), patch.object(comparison,
                'verified_summary', side_effect=lambda path, arm, suite, *args: self.reports[suite][arm]):
            return comparison.compare(self.results, self.plans, self.locks, self.root/'comparison')

    def test_report_contains_5_physics_3_vbench_totals_and_all_16_raw_and_normalized_dimensions(self):
        result = self.execute()
        self.assertTrue(result['paired_comparison_ready'])
        self.assertEqual(len(result['metrics']), 40)
        self.assertFalse(result['formal_benchmark_complete'])
        self.assertFalse(result['publication_eligible'])
        self.assertTrue((self.root/'comparison/comparison.md').exists())

    def test_changed_protocol_environment_or_prompt_lock_invalidates_stale_table(self):
        original = copy.deepcopy(self.reports)
        for key in ('jobs_sha256', 'subset_lock_sha256', 'weight_lock_sha256'):
            self.reports = copy.deepcopy(original)
            self.execute()
            self.reports['videophy2']['base'][key] = 'changed'
            with self.assertRaises(ValueError):
                self.execute()
            self.assertFalse((self.root/'comparison/comparison.csv').exists())
        self.reports = copy.deepcopy(original)
        self.reports['vbench']['base']['generation']['protocol'] = {'changed': True}
        with self.assertRaises(ValueError):
            self.execute()

    def test_summary_requires_real_execution_receipt_and_unchanged_raw_scores(self):
        scores = self.root/'base/videophy2/scores'
        videos = scores.parent/'videos'
        generation = dict(group='base', checkpoint_sha256=None)
        write_json(videos/'generation_lock.json', generation)
        scores.mkdir(parents=True)
        for mode in ('sa', 'pc'):
            (scores/f'videophy2_{mode}.csv').touch()
        report = dict(suite='videophy2', model='base', complete=True, subset_table_ready=True,
                      main_table_ready=False, formal_benchmark_complete=False, denominator=100,
                      subset_lock_sha256=sha256(self.locks['videophy2']), generation=generation,
                      generation_lock_sha256=sha256(videos/'generation_lock.json'),
                      result_sha256={m: sha256(scores/f'videophy2_{m}.csv') for m in ('sa', 'pc')})
        write_json(scores/'summary.json', report)
        receipt = dict(summary_sha256=sha256(scores/'summary.json'), official_scorers_executed=True,
                       subset_table_ready=True, tasks=[dict(returncode=0), dict(returncode=0)])
        write_json(scores/'score_status.json', receipt)
        with patch.object(comparison, 'recorded_jobs', return_value=[{}]*100):
            comparison.verified_summary(scores/'summary.json', 'base', 'videophy2', [{}]*100,
                                        self.locks['videophy2'])
            (scores/'videophy2_sa.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'raw scores changed'):
                comparison.verified_summary(scores/'summary.json', 'base', 'videophy2', [{}]*100,
                                            self.locks['videophy2'])
            write_json(scores/'score_status.json', dict(receipt, official_scorers_executed=False))
            with self.assertRaisesRegex(ValueError, 'completed official scorer'):
                comparison.verified_summary(scores/'summary.json', 'base', 'videophy2', [{}]*100,
                                            self.locks['videophy2'])


class PairedQueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.args = argparse.Namespace(output=str(self.root/'output'), plans=str(self.root/'plans'),
            checkpoint=str(self.root/'checkpoint.pt'), c_existing=str(self.root/'C_existing'),
            videophy2_environment=str(self.root/'vp_env.json'), vbench_environment=str(self.root/'vb_env.json'),
            devices=[f'cuda:{i}' for i in range(4)], minimum_free_gib=50, config=None, publish=False)
        Path(self.args.checkpoint).touch()
        write_json(self.args.videophy2_environment, dict(fixture=True))
        write_json(Path(self.args.c_existing)/'subset_lock.json', dict(suite='videophy2', fixture=True))
        write_json(Path(self.args.c_existing)/'videophy2/scores/summary.json', dict(fixture=True))
        for suite in ('videophy2', 'vbench'):
            write_jsonl(Path(self.args.plans)/f'{suite}.jsonl', [dict(fixture=True)])
        self.visits = []

    def prepare(self, plan, lock):
        contract = dict(suite=plan.stem, fixture=True)
        write_json(lock, contract)
        return contract

    def execute(self, fail=False):
        original = dict(generation=dict(checkpoint_sha256=sha256(self.args.checkpoint)),
                        weight_lock_sha256=sha256(self.args.videophy2_environment), scores={})

        def generate(args, suite, state, persist):
            self.visits.append((args.group, suite, args.checkpoint, args.devices, args.subset_lock))
            state['stage'] = suite+'_generation'
            state['suites'][suite].update(generation_complete=True,
                completed_videos=state['suites'][suite]['expected_videos'])
            persist()

        def score_model(args, suite, state, persist):
            if fail:
                raise RuntimeError('Synthetic scorer failure')
            state['stage'] = suite+'_scoring'
            state['suites'][suite]['scoring_complete'] = True
            persist()

        with patch.object(queue, 'prepare', side_effect=self.prepare), patch.object(queue,
                'select', return_value=[{}]*100), patch.object(queue, 'verified_summary', return_value=original), patch.object(
                queue, 'generate_suite', side_effect=generate), patch.object(queue, 'score_suite', side_effect=score_model), patch.object(
                queue, 'compare') as compare_call:
            result = queue.run(self.args)
            return result, compare_call

    def test_same_four_gpus_and_locked_prompts_cover_both_models_and_both_suites(self):
        state, compare_call = self.execute()
        self.assertTrue(state['complete'])
        self.assertTrue(state['paired_comparison_ready'])
        self.assertEqual([(v[0], v[1]) for v in self.visits],
                         [('base', 'videophy2'), ('C', 'vbench'), ('base', 'vbench')])
        self.assertIsNone(self.visits[0][2])
        self.assertIsNone(self.visits[2][2])
        self.assertEqual(self.visits[1][2], self.args.checkpoint)
        self.assertTrue(all(v[3] == self.args.devices for v in self.visits))
        self.assertEqual(self.visits[1][4], self.visits[2][4])
        compare_call.assert_called_once()
        self.assertTrue(state['arms']['C']['suites']['videophy2']['reused'])

    def test_scorer_failure_never_marks_comparison_complete_or_publishes_table(self):
        with self.assertRaisesRegex(RuntimeError, 'Synthetic scorer failure'):
            self.execute(fail=True)
        state = json.loads((Path(self.args.output)/'status.json').read_text())
        self.assertFalse(state['complete'])
        self.assertFalse(state['paired_comparison_ready'])
        self.assertTrue(state['failed'])
        self.assertFalse((Path(self.args.output)/'comparison/comparison.csv').exists())


if __name__ == '__main__':
    unittest.main()
