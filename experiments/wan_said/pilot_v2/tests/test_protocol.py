import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.wan_said.alignment import Grounding, Projector, distillation_loss
from experiments.wan_said.data import audit_splits, load_split
from experiments.wan_said.io_utils import sha256, write_json
from experiments.wan_said.pilot_v2.benchmarks import (coverage, make_jobs, require_complete,
                                                     suites, validate_job_paths, validate_official_plan)
from experiments.wan_said.pilot_v2.config import load_config, signature
from experiments.wan_said.pilot_v2.execution import deterministic_attention
from experiments.wan_said.pilot_v2.inference import native_generation_protocol, solver
from experiments.wan_said.pilot_v2.online import OnlineFeatures
from experiments.wan_said.pilot_v2.review import approve
from experiments.wan_said.pilot_v2.train import require_review, sample_index, state_hash
from experiments.wan_said.pilot_v2.teacher import interpolate_positions, source_module
from experiments.wan_said.runtime import ROOT


class AlgorithmTests(unittest.TestCase):
    def test_vit_base_grounding_and_frozen_target_contract(self):
        torch.manual_seed(42)
        ground = Grounding(video_dim=768, dim=256).eval().requires_grad_(False)
        text, video = torch.randn(4, 4096), torch.randn(4, 768)
        full, said = ground.target(text, video, 'full'), ground.target(text, video, 'said')
        self.assertEqual(full.shape, (4, 256))
        self.assertFalse(full.requires_grad or said.requires_grad)
        self.assertFalse(torch.equal(full, said))
        hidden = torch.randn(4, 7, 1536, requires_grad=True)
        loss = distillation_loss(Projector(1536, 256)(hidden.mean(1)), said)
        loss.backward()
        self.assertGreater(hidden.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is None for p in ground.parameters()))

    def test_stateless_shuffle_has_full_coverage_and_resume_order(self):
        for epoch in (0, 1, 2):
            order = [sample_index(i, 41, 42) for i in range(epoch*41, (epoch+1)*41)]
            self.assertEqual(sorted(order), list(range(41)))
        whole = [sample_index(i, 41, 42) for i in range(100)]
        resumed = whole[:37]+[sample_index(i, 41, 42) for i in range(37, 100)]
        self.assertEqual(whole, resumed)
        self.assertNotEqual(whole[:41], whole[41:82])

    def test_state_roundtrip_hash_changes_after_optimizer_step(self):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters())
        initial = state_hash(model.state_dict())
        model(torch.ones(1, 4)).sum().backward()
        optimizer.step()
        self.assertNotEqual(initial, state_hash(model.state_dict()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.pt'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), path)
            saved = torch.load(path, weights_only=True)
            self.assertEqual(state_hash(saved['model']), state_hash(model.state_dict()))
            self.assertTrue(saved['optimizer']['state'])

    def test_human_review_cannot_approve_unknown_test_or_rejected_samples(self):
        identity = dict(preprocessing_hash='p', split_lock_sha256='s')
        review = dict(approved=True, reviewer='human', reviewed_sample_ids=['train-id'],
                      preprocessing_hash='p', split_lock_sha256='s', rejected_sample_ids=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'review.json'
            write_json(path, review)
            require_review(path, identity, {'train-id', 'val-id'})
            for update in ({'reviewed_sample_ids': ['test-id']}, {'rejected_sample_ids': ['train-id']},
                           {'approved': False}, {'preprocessing_hash': 'old'}):
                write_json(path, dict(review, **update))
                with self.assertRaises(ValueError):
                    require_review(path, identity, {'train-id', 'val-id'})


@unittest.skipUnless((ROOT/'datasets/pilot_v2/manifests/split_lock.json').exists(), 'Local frozen pilot required')
class FrozenPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config()

    def test_parent_membership_preserved_new_ids_and_no_local_window(self):
        cfg = self.cfg
        splits = {s: load_split(cfg['manifests'], s) for s in ('train', 'val', 'test')}
        audit_splits(splits)
        self.assertEqual({s: len(rows) for s, rows in splits.items()}, {'train': 3995, 'val': 498, 'test': 500})
        lock = json.loads((Path(cfg['manifests'])/'split_lock.json').read_text())
        self.assertEqual(lock['parent_split_lock_sha256'], sha256(Path(cfg['parent_manifests'])/'split_lock.json'))
        self.assertEqual(lock['preprocessing_hash'], signature(cfg))
        for split in ('train', 'val', 'test'):
            old = load_split(cfg['parent_manifests'], split)
            new = load_split(cfg['manifests'], split)
            self.assertEqual({r['sha256'] for r in old}, {r['sha256'] for r in new})
            self.assertTrue(all('window_start' not in r for r in new))
            self.assertTrue(all(r['sample_id'] != r['parent_sample_id'] for r in new))

    def test_online_encoders_reject_sealed_test_or_edited_caption_before_loading(self):
        provider = OnlineFeatures(self.cfg)
        row = load_split(self.cfg['manifests'], 'test')[0]
        with self.assertRaises(ValueError):
            provider(row)
        row = dict(load_split(self.cfg['manifests'], 'train')[0], caption='changed caption')
        with self.assertRaises(ValueError):
            provider(row)
        self.assertIsNone(provider.text)
        self.assertIsNone(provider.vae)
        self.assertIsNone(provider.teacher)

    def test_review_requires_completed_named_decisions_and_records_rejection(self):
        cfg = self.cfg
        row = load_split(cfg['manifests'], 'train')[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = dict(preprocessing_hash=signature(cfg),
                         split_lock_sha256=sha256(Path(cfg['manifests'])/'split_lock.json'),
                         samples=[dict(sample_id=row['sample_id'], preview='fixture.jpg')])
            write_json(root/'audit.json', audit)
            decision = dict(sample_id=row['sample_id'], split='train', caption=row['caption'],
                            preview='fixture.jpg', event_coverage='yes', caption_match='yes',
                            crop_preserves_subject='no', approve='no',
                            rejection_reason='fixture: crop loses subject', reviewer='test-fixture-only')

            def sheet(item):
                with (root/'review.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(item))
                    writer.writeheader()
                    writer.writerow(item)

            sheet(decision)
            result = approve(cfg, root/'review.csv', root/'audit.json', root/'rejected.json')
            self.assertFalse(result['approved'])
            self.assertEqual(result['rejected_sample_ids'], [row['sample_id']])
            for update in ({'approve': 'yes'}, {'reviewer': ''}, {'event_coverage': ''},
                           {'preview': 'wrong.jpg'}, {'rejection_reason': ''}):
                sheet(dict(decision, **update))
                with self.assertRaises(ValueError):
                    approve(cfg, root/'review.csv', root/'audit.json', root/'invalid.json')


@unittest.skipUnless((ROOT/'vendor/Wan2.1/wan/utils/fm_solvers_unipc.py').exists(), 'Pinned Wan source required')
class OfficialSolverTests(unittest.TestCase):
    def test_deterministic_backend_scope_restores_native_reference_on_error(self):
        from experiments.wan_said.runtime import forward_fn, upstream
        upstream()
        from diffsynth.models import wan_video_dit as native
        before = native.flash_attn if native.FLASH_ATTN_2_AVAILABLE else None
        forward = forward_fn()
        enabled = torch.are_deterministic_algorithms_enabled()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with deterministic_attention():
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertIs(forward_fn(), forward)
                if before is not None:
                    self.assertIs(native.flash_attn.flash_attn_func.func, before.flash_attn_func)
                    self.assertTrue(native.flash_attn.flash_attn_func.keywords['deterministic'])
                raise RuntimeError('intentional')
        if before is not None:
            self.assertIs(native.flash_attn, before)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), enabled)

    def test_parameterized_positions_match_official_48_frame_interpolator_exactly(self):
        source = source_module()
        original = torch.rand(1, 8*14*14, 12)
        dummy = SimpleNamespace(pos_embed=original.clone(), patch_embed=SimpleNamespace(tubelet_size=2))
        source.VisionTransformer.interpolate_pos_embed(dummy, 16, 224, 16, 12, target_resolution=[160, 240])
        ours = interpolate_positions(original, (8, 14, 14), (24, 10, 15))
        self.assertTrue(torch.equal(ours, dummy.pos_embed))

    def test_pinned_native_unipc_is_finite_and_reproducible(self):
        protocol = native_generation_protocol()
        self.assertEqual((protocol['frames'], protocol['height'], protocol['width'], protocol['fps']), (81, 480, 832, 16))
        self.assertEqual((protocol['steps'], protocol['cfg'], protocol['shift']), (50, 6., 8.))
        self.assertFalse(protocol['prompt_extension'])
        outputs = []
        for _ in range(2):
            schedule = solver(dict(protocol, steps=4), 'cpu')
            sample = torch.ones(1, 16, 5, 4, 4)
            for t in schedule.timesteps:
                sample = schedule.step(torch.full_like(sample, .2), t, sample, return_dict=False)[0]
            self.assertTrue(torch.isfinite(sample).all())
            outputs.append(sample)
        self.assertTrue(torch.equal(*outputs))


@unittest.skipUnless((ROOT/'vendor/VBench/vbench/VBench_full_info.json').exists(), 'Pinned benchmark sources required')
class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sets, cls.audit = suites()

    def test_official_prompt_counts_and_version_discrepancy(self):
        self.assertEqual(len(self.sets['t2v_compbench']), 1400)
        self.assertEqual(set(self.audit['t2v_compbench']['categories'].values()), {200})
        self.assertEqual([(r['category'], r['row']) for r in
                          self.audit['t2v_compbench']['normalized_prompt_metadata_differences']],
                         [('motion_binding', 195), ('action_binding', 159)])
        self.assertEqual(self.audit['vbench']['official_metadata_rows'], 946)
        self.assertEqual(len(self.sets['vbench']), 944)
        self.assertEqual(len(self.audit['vbench']['dimensions']), 16)
        self.assertEqual(self.audit['videophy2']['pinned_csv_rows'], 590)
        self.assertEqual(self.audit['videophy2']['alternate_csv_rows'], 591)
        self.assertFalse(self.audit['videophy2']['requested_344_protocol_verified'])

    def test_job_mapping_and_official_filenames(self):
        for suite, rows in self.sets.items():
            jobs = make_jobs(suite, rows)
            validate_job_paths(jobs)
            self.assertEqual(len(jobs), len(rows)*(5 if suite == 'vbench' else 1))
        first = self.sets['vbench'][0]
        self.assertEqual(make_jobs('vbench', [first])[0]['relative_path'], f'vbench/{first["prompt"]}-0.mp4')

    def test_incomplete_or_modified_plan_rejected_for_formal_scoring(self):
        jobs = make_jobs('videophy2', self.sets['videophy2'])
        validate_official_plan(jobs)
        for changed in ([], jobs[:344], [dict(jobs[0], seed=999)]+jobs[1:]):
            with self.assertRaises(ValueError):
                validate_official_plan(changed)

    def test_missing_videos_never_reduce_denominator(self):
        jobs = make_jobs('videophy2', self.sets['videophy2'])
        with tempfile.TemporaryDirectory() as tmp:
            result = coverage(jobs, tmp)
            self.assertEqual(result['denominator'], 590)
            self.assertEqual(result['available'], 0)
            with self.assertRaises(ValueError):
                require_complete(jobs, tmp)

    def test_duplicate_and_unsafe_mappings_rejected(self):
        job = make_jobs('videophy2', self.sets['videophy2'][:1])[0]
        for changed in ([job, job], [dict(job, relative_path='../escape.mp4')],
                        [dict(job, relative_path='/absolute.mp4')], [dict(job, job_id='../escape')]):
            with self.assertRaises(ValueError):
                validate_job_paths(changed)


if __name__ == '__main__':
    unittest.main()
