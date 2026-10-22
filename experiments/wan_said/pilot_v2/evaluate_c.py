"""Resumable full-plan C generation and official evaluation, independent of training."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ..io_utils import read_jsonl, sha256, write_json
from .benchmarks import validate_official_plan
from .inference import native_generation_protocol, verify_video

MODULE = 'experiments.wan_said.pilot_v2'
REPO = Path(__file__).resolve().parents[3]


def recorded_jobs(jobs, videos, verify=False):
    done = []
    for job in jobs:
        record = videos/'records'/(job['job_id']+'.json')
        path = videos/job['relative_path']
        if not record.exists():
            continue
        saved = json.loads(record.read_text())
        if saved['job'] != job or not path.is_file():
            raise ValueError('Generated record belongs to a different job or lacks its video')
        if verify:
            if saved['sha256'] != sha256(path):
                raise ValueError('Generated video checksum changed')
            verify_video(path, native_generation_protocol())
        done.append(saved)
    return done


def generate_suite(args, suite, state, persist):
    plan = Path(args.plans)/f'{suite}.jsonl'
    jobs = read_jsonl(plan)
    validate_official_plan(jobs)
    if getattr(args, 'subset_lock', None):
        from .subset import select
        jobs = select(jobs, plan, args.subset_lock)
    videos = Path(args.output)/suite/'videos'
    videos.mkdir(parents=True, exist_ok=True)
    state['stage'] = f'{suite}_generation'
    stage = state['suites'][suite]
    processes = []
    with contextlib.ExitStack() as stack:
        try:
            for index, device in enumerate(args.devices):
                log = stack.enter_context((videos/f'worker_{index}.log').open('a'))
                argv = [sys.executable, '-u', '-m', MODULE+'.inference', '--jobs', str(plan),
                        '--output', str(videos), '--group', 'C', '--checkpoint', args.checkpoint,
                        '--exploratory', '--worker-index', str(index), '--worker-count', str(len(args.devices)),
                        '--device', device, '--minimum-free-gib', str(args.minimum_free_gib)]
                if args.config:
                    argv.extend(['--config', args.config])
                if getattr(args, 'subset_lock', None):
                    argv.extend(['--subset-lock', str(args.subset_lock)])
                processes.append(subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
            stage['worker_pids'] = [p.pid for p in processes]
            while True:
                stage['completed_videos'] = len(recorded_jobs(jobs, videos))
                persist()
                codes = [p.poll() for p in processes]
                if any(c is not None and c != 0 for c in codes):
                    raise RuntimeError(f'{suite} generation worker failed: {codes}; inspect worker logs')
                if all(c is not None for c in codes):
                    break
                time.sleep(15)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    done = recorded_jobs(jobs, videos, verify=True)
    if len(done) != len(jobs):
        raise ValueError('Full generation coverage is required before official scoring')
    stage.update(completed_videos=len(done), generation_complete=True,
                 mean_generation_seconds=sum(r['seconds'] for r in done)/len(done))
    write_json(videos/'progress.json', dict(expected=len(jobs), denominator=len(jobs),
               completed=len(done), complete=True, smoke=False))
    persist()


def score_suite(args, suite, state, persist):
    root = Path(args.output)/suite
    environment = Path(getattr(args, suite+'_environment'))
    state['stage'] = f'{suite}_scoring'
    while not environment.is_file():
        state['stage'] = f'{suite}_waiting_for_scorer_environment'
        persist()
        time.sleep(60)
    state['stage'] = f'{suite}_scoring'
    state['suites'][suite]['scorer_environment_sha256'] = sha256(environment)
    persist()
    argv = [sys.executable, '-u', '-m', MODULE+'.score', '--suite', suite,
            '--jobs', str(Path(args.plans)/f'{suite}.jsonl'), '--videos', str(root/'videos'),
            '--output', str(root/'scores'), '--environments', str(environment)]
    if getattr(args, 'subset_lock', None):
        argv.extend(['--subset-lock', str(args.subset_lock)])
    previous = root/'scores/score_status.json'
    ready_key = 'subset_table_ready' if getattr(args, 'subset_lock', None) else 'main_table_ready'
    if previous.exists() and json.loads(previous.read_text()).get(ready_key):
        argv.append('--aggregate-only')
    with (root/'score_driver.log').open('a') as log:
        subprocess.run(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads((root/'scores/score_status.json').read_text())
    if not result.get(ready_key):
        raise RuntimeError('Official scorer did not produce a complete comparable summary')
    state['suites'][suite].update(scoring_complete=True, scores=result['scores'])
    persist()


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plans = {s: sha256(Path(args.plans)/f'{s}.jsonl') for s in ('videophy2', 'vbench')}
        source = Path(__file__).parent
        contract = dict(checkpoint_sha256=sha256(args.checkpoint), plans_sha256=plans,
                        protocol=native_generation_protocol(), group='C', exploratory=True,
                        publication_eligible=False, devices=args.devices, feature_cache_enabled=False,
                        source_sha256={n: sha256(source/n) for n in
                                       ('evaluate_c.py', 'inference.py', 'score.py', 'aggregate.py')})
        lock = output/'run_lock.json'
        if lock.exists() and json.loads(lock.read_text()) != contract:
            raise ValueError('Official queue identity changed; refusing to mix protocols/checkpoints')
        write_json(lock, contract)
        state = dict(complete=False, stage='starting', pid=os.getpid(), suites={})
        for suite in plans:
            state['suites'][suite] = dict(expected_videos=len(read_jsonl(Path(args.plans)/f'{suite}.jsonl')),
                                          completed_videos=0, generation_complete=False, scoring_complete=False)

        def persist():
            state['updated_unix'] = time.time()
            write_json(output/'status.json', state)

        try:
            for suite in plans:
                generate_suite(args, suite, state, persist)
                score_suite(args, suite, state, persist)
            state.update(complete=True, stage='complete')
        except BaseException as error:
            state.update(failed=True, error=repr(error))
            persist()
            raise
        persist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--plans', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--videophy2-environment', required=True)
    parser.add_argument('--vbench-environment', required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--minimum-free-gib', type=float, default=50)
    run(parser.parse_args())
