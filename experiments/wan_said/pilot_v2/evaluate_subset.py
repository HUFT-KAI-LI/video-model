"""Run exactly the requested 100-video C pilot with official VideoPhy-2 scoring."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from ..io_utils import read_jsonl, sha256, write_json
from .evaluate_c import generate_suite, score_suite
from .inference import native_generation_protocol, verify_video
from .subset import prepare, select


def reuse(jobs, source, videos, checkpoint, full_plan):
    source, videos = Path(source), Path(videos)
    original = json.loads((source/'generation_lock.json').read_text())
    if (original['group'] != 'C' or original['smoke'] or
            original['checkpoint_sha256'] != sha256(checkpoint) or
            original['jobs_sha256'] != sha256(full_plan) or
            original['protocol'] != native_generation_protocol()):
        raise ValueError('Cannot reuse videos from another adapter/plan/generation protocol')
    reused = []
    for job in jobs:
        record = source/'records'/(job['job_id']+'.json')
        target = videos/job['relative_path']
        if not record.exists() or target.exists():
            continue
        saved = json.loads(record.read_text())
        path = source/job['relative_path']
        if saved['job'] != job or sha256(path) != saved['sha256']:
            raise ValueError('Reused generated video identity or checksum changed')
        verify_video(path, original['protocol'])
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target)
        saved.update(path=str(target.resolve()), reused_from=str(path.resolve()),
                     source_record_sha256=sha256(record),
                     source_generation_lock_sha256=sha256(source/'generation_lock.json'))
        write_json(videos/'records'/record.name, saved)
        reused.append(job['job_id'])
    return reused


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    args.output = str(output)
    args.plans = str(Path(args.plans).resolve())
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.videophy2_environment = str(Path(args.videophy2_environment).resolve())
    full_plan = Path(args.plans)/'videophy2.jsonl'
    with (output/'queue.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.subset_lock = str(output/'subset_lock.json')
        selection = prepare(full_plan, args.subset_lock, args.count, args.selection_seed)
        jobs = select(read_jsonl(full_plan), full_plan, args.subset_lock)
        here = Path(__file__).parent
        contract = dict(checkpoint_sha256=sha256(args.checkpoint), subset_lock_sha256=sha256(args.subset_lock),
                        full_plan_sha256=sha256(full_plan), protocol=native_generation_protocol(),
                        evaluation_scope='subset_pilot', denominator=len(jobs), full_benchmark_denominator=590,
                        group='C', exploratory=True, publication_eligible=False, vbench_queued=False,
                        formal_benchmark_complete=False, feature_cache_enabled=False, devices=args.devices,
                        scorer_environment_sha256=sha256(args.videophy2_environment),
                        source_sha256={name: sha256(here/name) for name in
                            ('evaluate_subset.py', 'evaluate_c.py', 'subset.py', 'inference.py', 'score.py', 'aggregate.py')})
        lock = output/'run_lock.json'
        if lock.exists() and json.loads(lock.read_text()) != contract:
            raise ValueError('Frozen subset queue changed; refusing to mix checkpoints/selections/protocols')
        write_json(lock, contract)
        reused = reuse(jobs, args.source_videos, output/'videophy2/videos', args.checkpoint, full_plan)
        write_json(output/'reuse_report.json', dict(newly_reused_job_ids=reused, hardlinks=True,
                   selection_independent_of_existing_videos=True, subset_size=len(jobs),
                   full_benchmark_denominator=590, source_videos=args.source_videos))
        state = dict(complete=False, subset_complete=False, stage='starting', pid=os.getpid(),
                     evaluation_scope='subset_pilot', denominator=len(jobs), full_benchmark_denominator=590,
                     formal_benchmark_complete=False, publication_eligible=False, vbench_queued=False,
                     selection_seed=selection['selection_seed'], suites=dict(videophy2=dict(
                     expected_videos=len(jobs), completed_videos=0, generation_complete=False, scoring_complete=False)))

        def persist():
            state['updated_unix'] = time.time()
            write_json(output/'status.json', state)

        try:
            generate_suite(args, 'videophy2', state, persist)
            score_suite(args, 'videophy2', state, persist)
            state.update(complete=True, subset_complete=True, stage='subset_complete')
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
    parser.add_argument('--source-videos', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--selection-seed', type=int, default=20261022)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--minimum-free-gib', type=float, default=50)
    run(parser.parse_args())
