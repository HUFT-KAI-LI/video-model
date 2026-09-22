import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import REVIEW_FIELDS, build_plan, geometry, load_config, load_prompts, write_csv, write_json
from detect_candidates import score_video, select_pool
from extract_features import sample_indices, score_features
from summarize import detector_metrics, prefix_status, rates, snapshot_verified, validate_review


class ProtocolTests(unittest.TestCase):
    def test_prompt_design(self):
        prompts = load_prompts(HERE/'prompts.jsonl')
        self.assertEqual(Counter(p['difficulty'] for p in prompts), dict(easy=8, medium=8, stress=4))
        self.assertEqual(set(Counter(p['category'] for p in prompts).values()), {5})
        plan = build_plan(load_config(HERE/'generation_config.yaml'), prompts)
        self.assertEqual(len({r['video_id'] for r in plan}), 80)

    def test_block_mapping_covers_frames_without_overlap(self):
        for seconds in (45,60):
            g = geometry(seconds)
            covered = [f for b in g['blocks'] for f in range(b['frame_start'],b['frame_end'])]
            self.assertEqual(covered,list(range(seconds*16)))
            self.assertEqual(g['blocks'][0]['frame_end'],9)
            self.assertEqual(g['blocks'][1]['frame_start'],9)
            self.assertEqual(g['blocks'][-1]['end_sec'],seconds)
            self.assertEqual(g['latent_frames']%3,0)
        self.assertEqual(geometry(60)['latent_frames'],243)
        self.assertEqual(geometry(45)['latent_frames'],183)

    def test_samples_stay_in_block_including_trimmed_tail(self):
        blocks = geometry(60)['blocks']
        for b,indices in sample_indices(blocks,3).items():
            self.assertTrue(all(blocks[b]['frame_start']<=i<blocks[b]['frame_end'] for i in indices))
        self.assertEqual(len(sample_indices(blocks,3)[80]),3)

    def test_dino_reference_and_clip_margin(self):
        import numpy as np
        s,a = score_features([[2,0],[1,0],[0,1]],[.4,.3,.1],[[.1,.2],[.2,.1],[.3,.2]],[0,0,1],1)
        np.testing.assert_allclose(s,[1,1,0])
        np.testing.assert_allclose(a,[.2,.1,-.2])

    def scores(self,values):
        return [dict(block_index=b,subject=v,attribute=v) for b,v in enumerate(values)]

    def test_sustained_drop_excludes_single_frame_event(self):
        cfg=load_config(HERE/'generation_config.yaml')['mining']
        isolated=score_video(self.scores([1]*5+[.2]+[1]*8),cfg)
        persistent=score_video(self.scores([1]*5+[.2]*9),cfg)
        self.assertEqual(isolated['subject_drop'],0)
        self.assertAlmostEqual(persistent['subject_drop'],.8)
        self.assertEqual(persistent['subject_onset'],5)
        self.assertEqual(score_video(self.scores([.5]*14),cfg)['subject_drop'],0)
        with self.assertRaises(ValueError):
            score_video(self.scores([1]*10)+[dict(block_index=9,subject=1,attribute=1)],cfg)

    def test_pool_ties_and_no_fixed_absolute_cutoff(self):
        cfg=load_config(HERE/'generation_config.yaml')['mining']
        rows=[dict(video_id=str(i),**score_video(self.scores([1]*5+[v]*9),cfg)) for i,v in enumerate([.2,.2,1,1,1])]
        self.assertEqual(select_pool(rows,.2),{'0','1'})

    def review(self,**kwargs):
        r=dict.fromkeys(REVIEW_FIELDS,'')
        r.update(video_id='P001_S00',**kwargs)
        return r

    def test_human_validation_and_prefix_bounds(self):
        v=dict(video_id='P001_S00',duration_sec=60,fps=16)
        blocks=geometry(60)['blocks']
        blank=validate_review(self.review(),v,blocks)
        self.assertEqual(prefix_status(blank,60),'unknown')
        negative=validate_review(self.review(failure_confirmed='NO',reviewed_until_sec='30',human_confidence='high'),v,blocks)
        self.assertEqual(prefix_status(negative,30),'normal')
        self.assertEqual(prefix_status(negative,45),'unknown')
        # block 20 spans [14.8125,15.5625), straddling 15 seconds.
        positive=validate_review(self.review(failure_confirmed='YES',failure_type='identity_drift',
                       failure_onset_block='20',onset_kind='abrupt',pre_onset_normal='YES',
                       reviewed_until_sec='60',human_confidence='high'),v,blocks)
        self.assertEqual(prefix_status(positive,15),'unknown')
        self.assertEqual(prefix_status(positive,30),'failure')
        result=rates([v],{v['video_id']:positive},[15,30])
        self.assertIsNone(result[0]['failure_rate'])
        self.assertEqual((result[0]['lower_bound'],result[0]['upper_bound']),(0,1))
        self.assertEqual(result[1]['failure_rate'],1)

    def test_exact_event_at_T_not_before_T(self):
        v=dict(video_id='x',duration_sec=60,fps=16)
        r=validate_review(self.review(failure_confirmed='YES',failure_type='identity_drift',
                          failure_onset_block='20',failure_onset_frame='240',onset_kind='abrupt',
                          pre_onset_normal='YES',reviewed_until_sec='60',human_confidence='high'),v,geometry(60)['blocks'])
        self.assertEqual(prefix_status(r,15),'normal')
        self.assertEqual(prefix_status(r,30),'failure')

    def test_invalid_labels_fail_closed(self):
        v=dict(video_id='x',duration_sec=60,fps=16)
        for r in [self.review(failure_confirmed='NO'), self.review(failure_confirmed='YES',failure_type='made_up'),
                  self.review(failure_confirmed='NO',reviewed_until_sec='NaN',human_confidence='high')]:
            with self.assertRaises(ValueError):
                validate_review(r,v,geometry(60)['blocks'])

    def test_snapshot_requires_verified_matching_artifacts(self):
        from common import digest
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            video=dict(video_id='v',artifact_sha256={'replay.pt':'original'})
            self.assertEqual(snapshot_verified(run,video,3,'plan'),(False,''))
            path=run/'outputs/v/snapshot_after_003.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'test snapshot, not a model')
            meta=dict(verified=True,video_id='v',after_block=3,plan_signature='plan',
                      snapshot_sha256=digest(path),replay_sha256='original')
            write_json(path.with_suffix('.json'),meta)
            self.assertTrue(snapshot_verified(run,video,3,'plan')[0])
            path.write_bytes(b'corrupted')
            self.assertFalse(snapshot_verified(run,video,3,'plan')[0])
            self.assertFalse(snapshot_verified(run,video,-1,'plan')[0])

    def test_recall_is_not_claimed_from_candidates_only(self):
        vs=[dict(video_id='a',duration_sec=60),dict(video_id='b',duration_sec=60)]
        rs={'a':dict(failure_confirmed='YES'),'b':dict(failure_confirmed='',reviewed_until_sec=0)}
        cs=[dict(video_id=v['video_id'],selection='candidate' if i==0 else 'unselected',selected_subject='True' if i==0 else 'False',selected_subject_change='False',selected_attribute='False',selected_attribute_change='False') for i,v in enumerate(vs)]
        result=detector_metrics(vs,rs,cs)['combined']
        self.assertEqual(result['precision'],1)
        self.assertIsNone(result['recall'])
        rs['b']['failure_confirmed']='YES'
        self.assertEqual(detector_metrics(vs,rs,cs)['combined']['recall'],.5)

    def test_mining_review_report_end_to_end(self):
        from common import digest
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            subprocess.run([sys.executable,str(HERE/'generate_pool.py'),'--run-dir',directory],check=True,capture_output=True)
            plan=json.loads((run/'plan.json').read_text())
            videos=plan['videos'][:4]
            scores=[]
            for i,v in enumerate(videos):
                out=run/'outputs'/v['video_id']
                write_json(out/'complete.json',dict(v,plan_signature=plan['signature'],artifact_sha256={}))
                for block in geometry(60)['blocks']:
                    value=.9 if i or block['block_index']<24 else .2
                    scores.append(dict(video_id=v['video_id'],block_index=block['block_index'],
                                       start_sec=block['start_sec'],end_sec=block['end_sec'],subject=value,attribute=value))
            path=run/'manifests/block_scores.csv'
            write_csv(path,scores)
            write_json(path.with_suffix('.provenance.json'),dict(block_scores_sha256=digest(path)))
            subprocess.run([sys.executable,str(HERE/'detect_candidates.py'),'--run-dir',directory],check=True,capture_output=True)
            self.assertTrue((run/'review.html').exists())
            summary_command=[sys.executable,str(HERE/'summarize.py'),'--run-dir',directory,'--no-plots']
            subprocess.run(summary_command,check=True,capture_output=True)
            initial=json.loads((run/'manifests/summary.json').read_text())
            self.assertEqual(initial['prefix_rates'][-1]['unknown'],4)
            reviews=[]
            for i,v in enumerate(videos):
                r=self.review(failure_confirmed='YES' if i==0 else 'NO',reviewed_until_sec='60',human_confidence='high')
                r['video_id']=v['video_id']
                if i==0:
                    r.update(failure_type='clothing_color_drift',failure_onset_block='24',
                             failure_onset_frame='290',onset_kind='abrupt',pre_onset_normal='YES')
                reviews.append(r)
            write_csv(run/'human_review.csv',reviews,REVIEW_FIELDS)
            subprocess.run(summary_command,check=True,capture_output=True)
            report=json.loads((run/'manifests/summary.json').read_text())
            self.assertEqual(report['prefix_rates'][0]['failure_rate'],0)
            self.assertEqual(report['prefix_rates'][-1]['failure_rate'],.25)
            self.assertEqual(report['p0b_eligible'],0)
            self.assertEqual(report['controls_matched'],1)
            self.assertEqual(report['missing_generation'],76)
            self.assertEqual(report['detector']['combined']['recall'],1)
            if importlib.util.find_spec('matplotlib'):
                subprocess.run(summary_command[:-1],check=True,capture_output=True)
                self.assertEqual(len(list((run/'figures').glob('*.png'))),6)

    def test_prepare_cli_needs_no_gpu_and_prevents_changed_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            command=[sys.executable,str(HERE/'generate_pool.py'),'--run-dir',directory]
            subprocess.run(command,check=True,capture_output=True)
            plan=json.loads((Path(directory)/'plan.json').read_text())
            self.assertEqual(len(plan['videos']),80)
            self.assertFalse((Path(directory)/'outputs').exists())
            subprocess.run(command,check=True,capture_output=True)
            plan['config']['duration_sec']=45
            write_json(Path(directory)/'plan.json',plan)
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires existing LongLive torch environment')
class UpstreamObservationTests(unittest.TestCase):
    def test_observer_and_skip_decode_preserve_trajectory_and_rng(self):
        import importlib.util
        import types
        from unittest.mock import patch
        import torch
        from types import SimpleNamespace as NS
        root=HERE.parents[1]/'restream_mvp/code/LongLive'
        stubs={'utils.wan_wrapper':types.ModuleType('utils.wan_wrapper'),
               'utils.memory':types.ModuleType('utils.memory'),
               'utils.debug_option':types.ModuleType('utils.debug_option')}
        for n in ('WanDiffusionWrapper','WanTextEncoder','WanVAEWrapper'):
            setattr(stubs['utils.wan_wrapper'],n,None)
        for n in ('gpu','get_cuda_free_memory_gb','DynamicSwapInstaller','move_model_to_device_with_memory_preservation','log_gpu_memory'):
            setattr(stubs['utils.memory'],n,None)
        stubs['utils.debug_option'].DEBUG=False
        with patch.dict(sys.modules,stubs):
            spec=importlib.util.spec_from_file_location('p0a_test_pipeline',root/'pipeline/causal_inference.py')
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        class Scheduler:
            def add_noise(self,clean,noise,t):
                return clean+noise*.1
        class Generator(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.model=torch.nn.Identity()
            def get_scheduler(self): return Scheduler()
            def forward(self,noisy_image_or_video,**kw):
                cache=kw['kv_cache'][0]
                old=cache['k'].clone()
                cache['k'].copy_(noisy_image_or_video.mean())
                return None,noisy_image_or_video*.9+old.mean()*.01
        class Text(torch.nn.Module):
            def forward(self,**kw): return {}
        class VAE(torch.nn.Module):
            def decode_to_pixel(self,x,**kw): return x
        args=NS(denoising_step_list=[1000,500],warp_denoising_step=False,num_frame_per_block=3,
                model_kwargs=NS(local_attn_size=3),context_noise=0)
        pipe=mod.CausalInferencePipeline(args,'cpu',Generator(),Text(),VAE())
        pipe.frame_seq_length=1;pipe.num_transformer_blocks=1
        torch.manual_seed(10)
        noise=torch.randn(1,9,1,2,2)
        before=torch.get_rng_state()
        _,expected=pipe.inference(noise,['test'],return_latents=True)
        expected_rng=torch.get_rng_state()
        torch.set_rng_state(before)
        seen=[]
        def observe(b,prefix,pipeline):
            seen.append((b,prefix.clone(),pipeline.kv_cache1[0]['k'].clone()))
        actual=pipe.inference(noise,['test'],decode_video=False,block_callback=observe)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        self.assertEqual([r[0] for r in seen],[0,1,2])
        for b,prefix,cache in seen:
            torch.testing.assert_close(prefix,expected[:,:3*(b+1)],rtol=0,atol=0)
            self.assertTrue(torch.isfinite(cache).all())

    def test_cached_vae_decode_matches_full_decode(self):
        import torch
        spec=importlib.util.spec_from_file_location('p0a_test_vae',HERE.parents[1]/'restream_mvp/code/LongLive/wan/modules/vae.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        torch.set_num_threads(2)
        torch.manual_seed(12)
        vae=mod.WanVAE_(dim=4,z_dim=2,num_res_blocks=1,temperal_downsample=[False,True,True]).eval()
        z=torch.randn(1,2,6,2,2)
        with torch.inference_mode():
            full=vae.decode(z,[0.,1.])
            vae.clear_cache()
            first=vae.cached_decode(z[:,:,:3],[0.,1.])
            second=vae.cached_decode(z[:,:,3:],[0.,1.])
        self.assertEqual(first.shape[2],9)
        self.assertEqual(second.shape[2],12)
        torch.testing.assert_close(torch.cat([first,second],dim=2),full,rtol=1e-4,atol=1e-5)


if __name__=='__main__':
    unittest.main()
