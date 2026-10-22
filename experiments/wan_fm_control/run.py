"""Durable FM training, matched evaluation, and restoration of the existing queue."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

from ..wan_said.data import load_split
from ..wan_said.io_utils import read_jsonl, sha256, write_json
from ..wan_said.runtime import ROOT
from ..wan_said.pilot_v2.config import HERE, load_config
from ..wan_said.pilot_v2.evaluate_c import generate_suite, score_suite
from ..wan_said.pilot_v2.subset import select
from .audit import archive
from .compare import compare
from .preflight import resume_equivalence
from .train import check_reference

MODULE = 'experiments.wan_fm_control.'
REPO = ROOT.parents[1]


def recover(output, service):
    """Also called by ExecStopPost, including after termination or training failure."""
    receipt = Path(output)/'paused_queue.json'
    if not receipt.exists():
        return
    data = json.loads(receipt.read_text())
    if data['service'] != service:
        raise ValueError('Refusing to restore an unrelated service')
    status = json.loads(Path(data['status_path']).read_text())
    if data['was_active'] and not status.get('complete'):
        subprocess.run(['systemctl', '--user', 'start', service], check=True)


def matching_training(a, c):
    if (a['initial_lora_sha256'] != c['initial_lora_sha256'] or
            a['steps'] != c['steps'] or a['contract']['identity'] != c['contract']['identity']):
        raise ValueError('Initial adapter, optimizer budget or training identity differs')
    keys = ('sample_id', 'step', 'rank', 'noise_seed', 'timestep', 'sigma', 'sampling')
    if ([{k: r[k] for k in keys} for r in a['records']] !=
            [{k: r[k] for k in keys} for r in c['records']]):
        raise ValueError('A/C sample order, diffusion noise schedule or video preprocessing differs')
    return dict(complete=True, initial_lora_equal=True, training_identity_equal=True,
        ordered_samples_noise_timesteps_crops_equal=True, steps=a['steps'],
        unique_training_samples=len({r['sample_id'] for r in a['records']}),
        A_loss='FM', C_loss='FM + 0.05 * SAID', full_parameter_finetuning=False)


def run(args):
    output, reference_root, paired = map(lambda p: Path(p).resolve(),
                                        (args.output, args.reference_root, args.paired_root))
    output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = load_config(reference_root/'resolved_config.json')
        reference = reference_root/'C/checkpoint.pt'
        plans = {s: reference_root/'plans'/f'{s}.jsonl' for s in ('videophy2', 'vbench')}
        locks = {s: paired/'plans'/f'{s}_subset.json' for s in plans}
        env_root = Path(args.environment_root).resolve()
        environments = {s: env_root/f'{s}_environment.json' for s in plans}
        existing = json.loads((paired/'run_lock.json').read_text())
        shared = {str(HERE/name): digest for name, digest in existing['source_sha256'].items()}
        shared.update({str(ROOT/name): sha256(ROOT/name) for name in
                       ('runtime.py', 'alignment.py', 'data.py', 'io_utils.py', 'train.py')})
        own = {str(p): sha256(p) for p in Path(__file__).parent.glob('*.py')}
        files = {**shared, **own, str(reference): sha256(reference),
                 str(reference_root/'resolved_config.json'): sha256(reference_root/'resolved_config.json')}
        for p in [*plans.values(), *locks.values(), *environments.values(),
                  Path(args.c_existing).resolve()/'videophy2/scores/summary.json',
                  Path(cfg['manifests'])/'split_lock.json']:
            files[str(p)] = sha256(p)
        contract = dict(arm='A', loss='native FM only', training_samples=len(load_split(cfg['manifests'], 'train', True)),
            validation_samples=len(load_split(cfg['manifests'], 'val')), global_batch=4, epochs=1,
            optimizer_steps=math.ceil(len(load_split(cfg['manifests'], 'train', True))/4),
            source_and_input_sha256=files, parent_paired_lock_sha256=sha256(paired/'run_lock.json'),
            devices=args.devices, prior_queue_service=args.paired_service,
            feature_cache_bytes=0, test_used=False, publication_eligible=False)
        path = output/'run_lock.json'
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError('FM control run identity changed; use a new output directory')
        write_json(path, contract)
        config_path = output/'resolved_config.json'
        write_json(config_path, cfg)
        previous = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
        state = dict(complete=False, stage='preflight', started_unix=previous.get('started_unix', time.time()),
            training_complete=False, validation_complete=False,
            suites={s: dict(expected_videos=n, completed_videos=0, scoring_complete=False)
                    for s, n in dict(videophy2=100, vbench=500).items()})

        def persist():
            state['updated_unix'] = time.time()
            write_json(output/'status.json', state)

        def frozen():
            for name, digest in files.items():
                if sha256(name) != digest:
                    raise ValueError(f'Frozen FM control input/source changed: {name}')
            if sha256(paired/'run_lock.json') != contract['parent_paired_lock_sha256']:
                raise ValueError('The parent matched evaluation identity changed')

        def execute(stage, module, arguments, distributed=False):
            frozen()
            state['stage'] = stage
            persist()
            command = [sys.executable, '-u', '-m', MODULE+module, '--config', str(config_path), *map(str, arguments)]
            if distributed:
                command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                           '-m', MODULE+module, '--config', str(config_path), *map(str, arguments)]
            with (output/(stage+'.log')).open('a') as log:
                log.write(json.dumps(dict(command=command, started_unix=time.time()))+'\n')
                log.flush()
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)

        def evaluate(suite):
            frozen()
            args_eval = argparse.Namespace(config=str(config_path), group='A', exploratory=True,
                checkpoint=str(output/'A/checkpoint.pt'), plans=str(reference_root/'plans'),
                output=str(output/'evaluation'), devices=args.devices,
                minimum_free_gib=50, score_devices=[d.removeprefix('cuda:') for d in args.devices],
                subset_lock=str(locks[suite]), **{s+'_environment': str(p) for s, p in environments.items()})
            generate_suite(args_eval, suite, state, persist)
            frozen()
            score_suite(args_eval, suite, state, persist)

        sources = {s: dict(base=str(paired/'base'/s/'scores/summary.json'),
                           A=str(output/'evaluation'/s/'scores/summary.json'),
                           C=str(Path(args.c_existing).resolve()/'videophy2/scores/summary.json') if s == 'videophy2'
                           else str(paired/'C'/s/'scores/summary.json')) for s in plans}
        try:
            frozen()
            if shutil.disk_usage(output).free < 50*2**30:
                raise OSError('Retaining the 50 GiB free-disk safety margin')
            for suite in plans:
                jobs = select(read_jsonl(plans[suite]), plans[suite], locks[suite])
                if len(jobs) != (100 if suite == 'videophy2' else 500):
                    raise ValueError('Existing 100-prompt selection changed')
            c = torch.load(reference, map_location='cpu', weights_only=True)
            from ..wan_said.pilot_v2.online import experiment_identity
            check_reference(c, cfg, experiment_identity(cfg), contract['training_samples'])
            del c
            receipt = output/'paused_queue.json'
            if not receipt.exists():
                active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.paired_service]).returncode == 0
                parent_status = json.loads((paired/'status.json').read_text())
                write_json(receipt, dict(service=args.paired_service, was_active=active,
                    status_path=str(paired/'status.json'), paused_unix=time.time(), original_status=parent_status,
                    reason='Temporary GPU handoff to matched FM-only training and VideoPhy-2; existing records retained'))
            state['stage'] = 'pausing_prior_generation'
            persist()
            subprocess.run(['systemctl', '--user', 'stop', args.paired_service], check=True)
            interrupted = []
            for arm in ('base', 'C'):
                for suite, plan in plans.items():
                    videos = paired/arm/suite/'videos'
                    for job in select(read_jsonl(plan), plan, locks[suite]):
                        video = videos/job['relative_path']
                        if video.exists() and not (videos/'records'/(job['job_id']+'.json')).exists():
                            target = paired/'interrupted_artifacts'/f'{arm}_{suite}_{job["job_id"]}_{time.time_ns()}.mp4'
                            target.parent.mkdir(parents=True, exist_ok=True)
                            video.rename(target)
                            interrupted.append(dict(original=str(video), preserved_as=str(target)))
            if interrupted:
                write_json(output/('interrupted_prior_videos_'+str(time.time_ns())+'.json'), interrupted)
            proof = output/'preflight_proof.json'
            if not proof.exists():
                for label, steps, resume in [('one', 1, None), ('resumed', 2, output/'preflight_one/checkpoint.pt'),
                                             ('fresh', 2, None)]:
                    destination = output/('preflight_one' if label == 'resumed' else 'preflight_'+label)
                    if (destination/'checkpoint.pt').exists():
                        saved = torch.load(destination/'checkpoint.pt', map_location='cpu', weights_only=True)
                        done = steps <= saved['steps'] <= 2 and saved['contract']['smoke']
                        del saved
                        if done:
                            continue
                    command = ['--reference', reference, '--output', destination, '--smoke-steps', steps]
                    if resume:
                        command += ['--resume', resume]
                    execute('preflight_'+label, 'train', command, True)
                resumed = torch.load(output/'preflight_one/checkpoint.pt', map_location='cpu', weights_only=True)
                fresh = torch.load(output/'preflight_fresh/checkpoint.pt', map_location='cpu', weights_only=True)
                if any(s['steps'] != 2 or not s['contract']['smoke'] or s['group'] != 'A' or
                       s['contract']['reference_C_sha256'] != files[str(reference)] for s in (resumed, fresh)):
                    raise ValueError('Preflight checkpoints are not two-update smoke runs from this reference')
                equivalence = resume_equivalence(resumed, fresh)
                write_json(proof, dict(complete=True, native_geometry=[cfg['frames'], cfg['height'], cfg['width']],
                    real_four_gpu_updates=2, **equivalence,
                    final_lora_sha256=fresh['final_lora_sha256'], gradient_audit=fresh['gradient_audit'],
                    teacher_used=False, feature_cache_bytes=0))
                del resumed, fresh
            checkpoint = output/'A/checkpoint.pt'
            command = ['--reference', reference, '--output', output/'A']
            done = False
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                done = saved['steps'] == contract['optimizer_steps'] and not saved['contract']['smoke']
                del saved
                command += ['--resume', checkpoint]
            if not done:
                execute('training_A', 'train', command, True)
            a, c = [torch.load(p, map_location='cpu', weights_only=True) for p in (checkpoint, reference)]
            write_json(output/'matched_training.json', matching_training(a, c))
            del a, c
            state['training_complete'] = True
            persist()
            execute('validation_A', 'validate', ['--checkpoint', checkpoint, '--output', output/'validation'])
            state['validation_complete'] = True
            persist()
            evaluate('videophy2')
            expected = dict(base=None, A=sha256(checkpoint), C=sha256(reference))
            compare(sources, plans, locks, output/'videophy2_comparison', ('videophy2',), expected)
            if args.publish:
                state['stage'] = 'archiving_videophy2'
                persist()
                state['videophy2_audit_commit'] = archive(output,
                    HERE/'reports'/('fm_control_videophy2_'+str(int(time.time()))), True)
            recover(output, args.paired_service)
            state['stage'] = 'waiting_prior_vbench_queue'
            persist()
            while True:
                parent = json.loads((paired/'status.json').read_text())
                active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', args.paired_service]).returncode == 0
                if not active:
                    if not parent.get('complete'):
                        raise RuntimeError('Prior Base/C queue stopped before finishing VBench; inspect its status/logs')
                    break
                time.sleep(60)
            evaluate('vbench')
            frozen()
            compare(sources, plans, locks, output/'comparison', expected_checkpoints=expected)
            write_json(output/'result_sources.json', sources)
            state.update(complete=True, stage='comparison_complete', completed_unix=time.time())
            persist()
            if args.publish:
                state['audit_commit'] = archive(output,
                    HERE/'reports'/('fm_control_complete_'+str(int(time.time()))), True)
                state['audit_uploaded'] = True
                persist()
        except BaseException as error:
            state.update(failed=True, error=repr(error))
            persist()
            raise
        finally:
            recover(output, args.paired_service)
        return state


def parser():
    p = argparse.ArgumentParser()
    runs = ROOT/'runs/pilot_v2'
    p.add_argument('--reference-root', default=str(runs/'c_validation_20261021'))
    p.add_argument('--paired-root', default=str(runs/'paired100_20261022'))
    p.add_argument('--c-existing', default=str(runs/'c_subset100_20261022'))
    p.add_argument('--environment-root', default=str(runs/'c_official_20261022/scorers'))
    p.add_argument('--output', required=True)
    p.add_argument('--paired-service', default='wan-said-paired100-20261022.service')
    p.add_argument('--devices', nargs=4, default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    p.add_argument('--publish', action='store_true')
    p.add_argument('--recover-only', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.recover_only:
        recover(args.output, args.paired_service)
    else:
        print(json.dumps(run(args), indent=2))
