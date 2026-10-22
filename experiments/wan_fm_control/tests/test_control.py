from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from experiments.wan_fm_control.compare import compare, matched, metric_rows
from experiments.wan_fm_control.run import matching_training, recover
from experiments.wan_fm_control.train import Objective, check_reference
from experiments.wan_said.pilot_v2.parallel.train import assignment
from experiments.wan_said.pilot_v2.train import sample_index


def training():
    return dict(initial_lora_sha256='same', steps=1, contract=dict(identity={'same': True}),
        records=[dict(sample_id='train1', step=1, rank=0, noise_seed=10042,
                      timestep=500., sigma=.5, sampling={'crop': [0, 0, 480, 720]})])


def summaries(suite='videophy2'):
    result = {}
    for i, arm in enumerate(('base', 'A', 'C')):
        result[arm] = dict(jobs_sha256='jobs', subset_lock_sha256='subset', weight_lock_sha256='scorer',
            generation=dict(base_weights={'w': 'sha'}, protocol={'steps': 50}),
            scores=dict(sa_mean=3.+i*.1, pc_mean=3.5+i*.1, sa_at_4=.3+i*.1,
                        pc_at_4=.6+i*.1, joint_at_4=.2+i*.1),
            paired_results=[dict(job_id=str(n), sa=3+(i > 0), pc=4) for n in range(100)])
        if suite == 'vbench':
            result[arm].update(scores=dict(quality_score=.7+i*.01, semantic_score=.6+i*.01, total_score=.68+i*.01),
                raw_dimensions={f'dim{n}': .5+i*.01 for n in range(16)},
                normalized_weighted_dimensions={f'dim{n}': .4+i*.01 for n in range(16)},
                coverage={f'dim{n}': dict(expected=25, scored=25) for n in range(16)},
                subset_metadata_sha256='metadata', prompt_count=100, samples_per_prompt=5)
    return result


class TrainingTests(unittest.TestCase):
    def test_matched_training_accepts_same_schedule(self):
        report = matching_training(training(), training())
        self.assertTrue(report['ordered_samples_noise_timesteps_crops_equal'])
        self.assertEqual(report['A_loss'], 'FM')

    def test_matched_training_rejects_initial_weights(self):
        a = training()
        a['initial_lora_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            matching_training(a, training())

    def test_matched_training_rejects_noise_sample_crop_or_time_change(self):
        for key in ('sample_id', 'noise_seed', 'sampling', 'timestep', 'rank'):
            with self.subTest(key=key):
                a = training()
                a['records'][0][key] = 'changed'
                with self.assertRaises(ValueError):
                    matching_training(a, training())

    def test_tail_padding_preserves_exact_three_example_average(self):
        values = [assignment(1, 7, 4, r) for r in range(4)]
        self.assertEqual([p for p, _ in values], [4, 5, 6, 6])
        self.assertEqual([s for _, s in values], [4/3, 4/3, 4/3, 0.])
        self.assertAlmostEqual(sum(s for _, s in values)/4, 1.)

    def reference(self):
        rows = [dict(sample_id=str(i)) for i in range(7)]
        cfg, identity = dict(seed=42, manifests='locked'), dict(dataset='locked')
        saved = dict(group='C', config=cfg, steps=2,
            contract=dict(smoke=False, identity=identity,
                parallel=dict(backend='torch_ddp', world_size=4, local_batch=1,
                              global_batch=4, tail='zero_weight_padding', epochs=1)),
            records=[dict(sample_id=rows[sample_index(i, 7, 42)]['sample_id'],
                          noise_seed=10042+i, rank=i % 4, step=i//4+1) for i in range(7)])
        return rows, cfg, identity, saved

    def test_reference_requires_full_matching_c_epoch(self):
        rows, cfg, identity, saved = self.reference()
        with patch('experiments.wan_fm_control.train.load_split', return_value=rows):
            check_reference(saved, cfg, identity, 7)
            saved['steps'] = 1
            with self.assertRaises(ValueError):
                check_reference(saved, cfg, identity, 7)

    def test_reference_rejects_different_batch(self):
        _, cfg, identity, saved = self.reference()
        saved['contract']['parallel']['global_batch'] = 1
        with self.assertRaises(ValueError):
            check_reference(saved, cfg, identity, 7)

    def test_reference_rejects_different_noise_order(self):
        rows, cfg, identity, saved = self.reference()
        saved['records'][0]['noise_seed'] += 1
        with patch('experiments.wan_fm_control.train.load_split', return_value=rows):
            with self.assertRaises(ValueError):
                check_reference(saved, cfg, identity, 7)

    def test_objective_calls_only_native_fm_no_teacher_or_projection(self):
        objective = Objective.__new__(Objective)
        torch.nn.Module.__init__(objective)
        objective.cfg, objective.device = dict(gradient_checkpointing=True), 'cpu'
        objective.dit = torch.nn.Linear(1, 1)
        objective.scheduler = SimpleNamespace(timesteps=torch.tensor([500.]), sigmas=torch.tensor([.5]))
        objective.latest = {}
        objective.native = lambda dit, latents, timestep, **kwargs: dit(latents)
        calls = []

        def native_loss(pipe, **kwargs):
            calls.append(kwargs)
            return pipe.model_fn(pipe.dit, kwargs['input_latents'], torch.tensor([500.])).square().mean()

        features = dict(text=dict(context=torch.ones(1, 1)), vae=dict(latents=torch.ones(1, 1)))
        with patch.dict(sys.modules, {'diffsynth.diffusion.loss': SimpleNamespace(FlowMatchSFTLoss=native_loss)}):
            with patch('torch.autocast', return_value=nullcontext()):
                loss = objective(features)
        loss.backward()
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(calls[0]), {'input_latents', 'context', 'use_gradient_checkpointing'})
        self.assertTrue(calls[0]['use_gradient_checkpointing'])
        self.assertEqual(list(dict(objective.named_children())), ['dit'])
        self.assertIsNotNone(objective.dit.weight.grad)
        self.assertEqual(objective.latest, dict(timestep=500., sigma=.5))


class ComparisonTests(unittest.TestCase):
    def test_three_arm_deltas(self):
        reports = summaries()
        matched(reports, 'videophy2')
        rows = metric_rows(reports, 'videophy2')
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(rows[0]['A_minus_base'], .1)
        self.assertAlmostEqual(rows[0]['C_minus_A'], .1)
        self.assertAlmostEqual(rows[0]['C_minus_base'], .2)

    def test_vbench_keeps_all_official_dimensions_and_aggregates(self):
        reports = summaries('vbench')
        matched(reports, 'vbench')
        self.assertEqual(len(metric_rows(reports, 'vbench')), 35)

    def test_mismatched_scoring_or_generation_rejected(self):
        for key in ('jobs_sha256', 'subset_lock_sha256', 'weight_lock_sha256'):
            reports = summaries()
            reports['A'][key] = 'wrong'
            with self.assertRaises(ValueError):
                matched(reports, 'videophy2')
        reports = summaries()
        reports['C']['generation']['protocol']['steps'] = 4
        with self.assertRaises(ValueError):
            matched(reports, 'videophy2')

    def test_vbench_five_seeds_and_metadata_required(self):
        for key, value in [('samples_per_prompt', 1), ('subset_metadata_sha256', 'wrong')]:
            reports = summaries('vbench')
            reports['A'][key] = value
            with self.assertRaises(ValueError):
                matched(reports, 'vbench')

    def test_missing_dimension_rejected(self):
        reports = summaries('vbench')
        del reports['A']['raw_dimensions']['dim1']
        with self.assertRaises(ValueError):
            metric_rows(reports, 'vbench')

    def test_partial_videophy_report_does_not_claim_all_protocols_complete(self):
        reports = summaries()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source.json'
            path.write_text('{}')
            sources = dict(videophy2={a: str(path) for a in reports})
            with patch('experiments.wan_fm_control.compare.read_jsonl', return_value=[]), \
                    patch('experiments.wan_fm_control.compare.select', return_value=[]), \
                    patch('experiments.wan_fm_control.compare.verified_summary',
                          side_effect=lambda path, arm, *rest: reports[arm]):
                result = compare(sources, dict(videophy2=path), dict(videophy2=path),
                                 Path(tmp)/'report', ('videophy2',))
            self.assertFalse(result['complete'])
            self.assertFalse(result['formal_benchmark_complete'])
            self.assertEqual(set(result['videophy2_paired_bootstrap']['intervals']),
                             {'A_minus_base', 'C_minus_A', 'C_minus_base'})
            self.assertTrue((Path(tmp)/'report/comparison.csv').exists())

    def test_failed_revalidation_removes_stale_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in ('csv', 'md', 'json'):
                (Path(tmp)/f'comparison.{suffix}').write_text('stale')
            with self.assertRaises(KeyError):
                compare({}, {}, {}, tmp)
            self.assertFalse((Path(tmp)/'comparison.csv').exists())

    def test_complete_report_requires_both_suites_and_has_40_metrics(self):
        reports = {s: summaries(s) for s in ('videophy2', 'vbench')}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source.json'
            path.write_text('{}')
            sources = {s: {a: str(path) for a in arms} for s, arms in reports.items()}
            with patch('experiments.wan_fm_control.compare.read_jsonl', return_value=[]), \
                    patch('experiments.wan_fm_control.compare.select', return_value=[]), \
                    patch('experiments.wan_fm_control.compare.verified_summary',
                          side_effect=lambda path, arm, suite, *rest: reports[suite][arm]):
                result = compare(sources, {s: path for s in reports}, {s: path for s in reports},
                                 Path(tmp)/'report')
            self.assertTrue(result['complete'])
            self.assertFalse(result['formal_benchmark_complete'])
            self.assertEqual(len(result['metrics']), 40)
            self.assertEqual(set(result['vbench_coverage']), {'base', 'A', 'C'})


class RecoveryTests(unittest.TestCase):
    def test_no_receipt_never_starts_service(self):
        with tempfile.TemporaryDirectory() as tmp, patch('experiments.wan_fm_control.run.subprocess.run') as call:
            recover(tmp, 'paired.service')
            call.assert_not_called()

    def recovery_case(self, complete, active, expected):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)/'parent.json'
            parent.write_text(json.dumps(dict(complete=complete)))
            (Path(tmp)/'paused_queue.json').write_text(json.dumps(dict(
                service='paired.service', was_active=active, status_path=str(parent))))
            with patch('experiments.wan_fm_control.run.subprocess.run') as call:
                recover(tmp, 'paired.service')
                self.assertEqual(call.call_count, expected)

    def test_failure_restores_incomplete_previously_active_queue(self):
        self.recovery_case(False, True, 1)

    def test_completed_queue_is_not_restarted(self):
        self.recovery_case(True, True, 0)

    def test_previously_inactive_queue_is_not_started(self):
        self.recovery_case(False, False, 0)

    def test_recovery_rejects_unrelated_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'paused_queue.json').write_text(json.dumps(dict(service='other.service')))
            with self.assertRaises(ValueError):
                recover(tmp, 'paired.service')


if __name__ == '__main__':
    unittest.main()
