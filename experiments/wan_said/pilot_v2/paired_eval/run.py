"""Four-GPU matched evaluation with explicit full-protocol 100-prompt subsets."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from ...io_utils import read_jsonl, sha256, write_json
from ..config import HERE
from ..evaluate_c import generate_suite, score_suite
from ..inference import native_generation_protocol
from ..subset import prepare, select
from .audit import publish, snapshot
from .compare import compare, verified_summary


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plans = {suite: Path(args.plans).resolve()/f'{suite}.jsonl' for suite in ('videophy2', 'vbench')}
        locks = {suite: output/'plans'/f'{suite}_subset.json' for suite in plans}
        selections = {suite: prepare(plans[suite], locks[suite]) for suite in plans}
        original_subset = Path(args.c_existing).resolve()/'subset_lock.json'
        if sha256(original_subset) != sha256(locks['videophy2']):
            raise ValueError('The paired pilot must reuse the identical original 100 VideoPhy-2 prompts')
        checkpoint = str(Path(args.checkpoint).resolve())
        vp_environment = str(Path(args.videophy2_environment).resolve())
        vb_environment = str(Path(args.vbench_environment).resolve())
        contract = dict(evaluation_scope='matched_100_prompt_subset', publication_eligible=False,
            formal_benchmark_complete=False, arms=['base', 'C'], checkpoint_C_sha256=sha256(checkpoint),
            native_protocol=native_generation_protocol(), feature_cache_enabled=False,
            devices=args.devices, plans_sha256={s: sha256(p) for s, p in plans.items()},
            subset_sha256={s: sha256(p) for s, p in locks.items()},
            videos_per_model=dict(videophy2=100, vbench=500), prompt_count_per_suite=100,
            videophy2_environment_sha256=sha256(vp_environment), vbench_environment_path=vb_environment,
            source_sha256={str(p.relative_to(HERE)): sha256(p) for p in HERE.rglob('*.py')
                if 'reports' not in p.parts and 'tests' not in p.parts and '__pycache__' not in p.parts
                and p.name != 'vbench_setup.py'})
        lock = output/'run_lock.json'
        if lock.exists() and json.loads(lock.read_text()) != contract:
            raise ValueError('Paired queue identity/source changed; refusing to mix models or protocols')
        write_json(lock, contract)
        c_summary = Path(args.c_existing).resolve()/'videophy2/scores/summary.json'
        c_jobs = select(read_jsonl(plans['videophy2']), plans['videophy2'], locks['videophy2'])
        original = verified_summary(c_summary, 'C', 'videophy2', c_jobs, locks['videophy2'])
        if (original['generation']['checkpoint_sha256'] != contract['checkpoint_C_sha256'] or
                original['weight_lock_sha256'] != sha256(vp_environment)):
            raise ValueError('Existing C result belongs to another adapter/scoring environment')
        write_json(output/'C_videophy2_reused.json', dict(source_summary=str(c_summary),
                   sha256=sha256(c_summary), generated_videos=100, scores=original['scores'],
                   raw_videos_copied=False, note='Previously verified identical 100-prompt C pilot reused'))
        previous = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
        state = dict(complete=False, paired_comparison_ready=False, stage='starting', pid=os.getpid(),
                     started_unix=previous.get('started_unix', time.time()),
                     evaluation_scope=contract['evaluation_scope'], formal_benchmark_complete=False,
                     publication_eligible=False, arms={arm: dict(suites={suite: dict(expected_videos=n,
                        completed_videos=0, generation_complete=False, scoring_complete=False)
                        for suite, n in contract['videos_per_model'].items()}) for arm in ('base', 'C')})
        state['arms']['C']['suites']['videophy2'].update(completed_videos=100,
            generation_complete=True, scoring_complete=True, reused=True, scores=original['scores'])

        def persist():
            state['updated_unix'] = time.time()
            write_json(output/'status.json', state)

        results = dict(videophy2=dict(C=str(c_summary)), vbench={})

        def check_source():
            if any(sha256(HERE/name) != value for name, value in contract['source_sha256'].items()):
                raise ValueError('Frozen evaluation source changed while the queue was running')

        try:
            for arm, suite in [('base', 'videophy2'), ('C', 'vbench'), ('base', 'vbench')]:
                check_source()
                current = argparse.Namespace(config=args.config, group=arm, exploratory=arm == 'C',
                    checkpoint=checkpoint if arm == 'C' else None, plans=str(Path(args.plans).resolve()),
                    output=str(output/arm), devices=args.devices, minimum_free_gib=args.minimum_free_gib,
                    score_devices=[d.removeprefix('cuda:') for d in args.devices],
                    subset_lock=str(locks[suite]), videophy2_environment=vp_environment,
                    vbench_environment=vb_environment)
                arm_state = state['arms'][arm]

                def persist_arm():
                    state['stage'] = arm+'_'+arm_state.get('stage', 'starting')
                    persist()

                generate_suite(current, suite, arm_state, persist_arm)
                check_source()
                score_suite(current, suite, arm_state, persist_arm)
                results[suite][arm] = str(output/arm/suite/'scores/summary.json')
                persist()
            write_json(output/'result_sources.json', results)
            check_source()
            compare(results, plans, locks, output/'comparison')
            state.update(complete=True, paired_comparison_ready=True, stage='comparison_complete',
                         completed_unix=time.time())
            persist()
        except BaseException as error:
            state.update(failed=True, error=repr(error))
            persist()
            raise
        if args.publish:
            state['stage'] = 'archiving_results'
            persist()
            destination = HERE/'reports'/('20261022_paired100_complete_'+str(int(time.time())))
            try:
                snapshot(output, destination)
                commit = publish(destination)
                state.update(stage='complete', audit_uploaded=True, audit_commit=commit)
            except Exception as error:
                state.update(stage='comparison_complete_audit_failed', audit_uploaded=False, audit_error=repr(error))
                persist()
                raise
            persist()
        return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--plans', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--c-existing', required=True)
    parser.add_argument('--videophy2-environment', required=True)
    parser.add_argument('--vbench-environment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--minimum-free-gib', type=float, default=50)
    parser.add_argument('--publish', action='store_true')
    print(json.dumps(run(parser.parse_args()), indent=2))
