"""Durable C pilot: validated Grounding -> full train pass -> held-out validation -> benchmarks."""
import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..data import load_split
from ..io_utils import sha256, write_json
from .config import HERE, load_config, signature

MODULE = 'experiments.wan_said.pilot_v2.'


def run(cfg, output, grounding_service, ground_steps=1000, exploratory=False, review=None,
        environments=None, benchmarks=('videophy2', 'vbench'), workers=1):
    if workers not in (1, 4):
        raise ValueError('Supported execution protocols are serial or four-rank DDP')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    training_count = len(load_split(cfg['manifests'], 'train', training=True))
    training_steps = math.ceil(training_count/workers)
    lock = dict(config=cfg, preprocessing_hash=signature(cfg),
                split_lock_sha256=sha256(Path(cfg['manifests'])/'split_lock.json'),
                source_sha256={str(p.relative_to(HERE)): sha256(p) for p in HERE.rglob('*.py')
                               if 'tests' not in p.parts and 'reports' not in p.parts},
                grounding_service=grounding_service, ground_steps=ground_steps,
                training_steps=training_steps, training_epochs=1, training_samples=training_count,
                workers=workers, global_batch=workers, validation_count=len(load_split(cfg['manifests'], 'val')),
                exploratory=exploratory, approved_review=str(Path(review).resolve()) if review else None,
                approved_review_sha256=sha256(review) if review else None,
                environments=str(Path(environments).resolve()) if environments else None,
                environments_sha256=sha256(environments) if environments else None,
                benchmarks=list(benchmarks), feature_cache_bytes=0, test_used=False)
    path = output/'run_lock.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError('Run identity changed; use a new run directory or restore the original code/config')
    write_json(path, lock)
    status = dict(lock_sha256=sha256(path), completed_stages=[], complete=False, official_scorers_executed=False,
                  publication_eligible=not exploratory, feature_cache_bytes=0, test_used=False)

    def update(stage, **extra):
        status.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **extra)
        write_json(output/'status.json', status)
        print(json.dumps(status), flush=True)

    def frozen():
        if (sha256(Path(cfg['manifests'])/'split_lock.json') != lock['split_lock_sha256'] or
                any(sha256(HERE/name) != value for name, value in lock['source_sha256'].items()) or
                (review and sha256(review) != lock['approved_review_sha256']) or
                (environments and sha256(environments) != lock['environments_sha256'])):
            raise ValueError('Dataset/code/review/evaluator lock changed during the experiment')

    config_path = output/'resolved_config.json'
    write_json(config_path, cfg)

    def execute(stage, module, arguments):
        frozen()
        update(stage)
        command = [sys.executable, '-m', MODULE+module, '--config', str(config_path), *map(str, arguments)]
        if module == 'parallel.train':
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                       '-m', MODULE+module, '--config', str(config_path), *map(str, arguments)]
        with (output/(stage+'.log')).open('a') as log:
            log.write(json.dumps(dict(command=command, started_at_utc=datetime.now(timezone.utc).isoformat()))+'\n')
            log.flush()
            subprocess.run(command, cwd=HERE.parents[2], stdout=log, stderr=subprocess.STDOUT, check=True)
        status['completed_stages'].append(stage)
        update(stage+'_complete')

    try:
        update('waiting_grounding')
        while True:
            result = subprocess.check_output(['systemctl', '--user', 'show', grounding_service,
                                               '--property=ActiveState', '--property=Result', '--property=MainPID'], text=True)
            properties = dict(line.split('=', 1) for line in result.splitlines() if '=' in line)
            report_path = output/'grounding/report.json'
            ground_report = json.loads(report_path.read_text()) if report_path.exists() else {}
            if (ground_report.get('steps') == ground_steps and properties.get('MainPID', '0') == '0' and
                    properties.get('Result', 'success') == 'success'):
                break
            if properties.get('ActiveState') in ('failed', 'inactive'):
                raise RuntimeError('Grounding service stopped before completing its registered steps; C will not start')
            if properties.get('ActiveState') not in ('active', 'activating', 'deactivating'):
                raise RuntimeError('Grounding service is absent or in an unknown state')
            time.sleep(30)
        import torch
        grounding_path = output/'grounding/grounding.pt'
        ground = torch.load(grounding_path, map_location='cpu', weights_only=True)
        if ground['steps'] != ground_steps or not ground['grounding_validated']:
            raise ValueError('Grounding did not complete the registered steps or pass held-out validation; C is blocked')
        status['completed_stages'].append('grounding')
        del ground
        checkpoint = output/'C/checkpoint.pt'
        if workers == 4:
            execute('parallel_preflight', 'parallel.probe',
                    ['--grounding', grounding_path, '--output', output])
        arguments = ['--group', 'C', '--steps', str(training_count), '--grounding', grounding_path,
                     '--output', output/'C', '--gradient-audit']
        if workers == 4:
            arguments = ['--grounding', grounding_path, '--output', output/'C', '--gradient-audit']
        if exploratory:
            arguments.append('--exploratory')
        if review:
            arguments += ['--approved-review', review]
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if (saved['contract']['grounding_sha256'] != sha256(grounding_path) or
                    bool(saved['contract'].get('exploratory')) != exploratory or saved['contract']['smoke'] or
                    saved['group'] != 'C' or saved['steps'] > training_steps or
                    saved['contract'].get('parallel', {}).get('world_size', 1) != workers):
                raise ValueError('Existing C checkpoint does not belong to this run')
            if saved['steps'] < training_steps:
                arguments += ['--resume', checkpoint]
            else:
                arguments = None
            del saved
        if arguments:
            execute('train_c', 'parallel.train' if workers == 4 else 'train', arguments)
        execute('validation', 'validate', ['--checkpoint', checkpoint, '--grounding', grounding_path,
                                           '--output', output/'validation'])
        execute('benchmark_plan', 'benchmarks', ['--output', output/'plans'])
        pending_suites = []
        for suite in benchmarks:
            arguments = ['--jobs', output/f'plans/{suite}.jsonl', '--group', 'C', '--checkpoint', checkpoint,
                         '--output', output/f'generation/{suite}']
            if exploratory:
                arguments.append('--exploratory')
            execute('generate_'+suite, 'inference', arguments)
            suite_environment = Path(environments) if environments else output/f'scorers/{suite}_environment.json'
            if suite_environment.is_file():
                # score does not take --config; invoke its pinned preflight directly in a child process.
                frozen()
                update('score_'+suite)
                command = [sys.executable, '-m', MODULE+'score', '--suite', suite, '--jobs', str(output/f'plans/{suite}.jsonl'),
                           '--videos', str(output/f'generation/{suite}'), '--output', str(output/f'scores/{suite}'),
                           '--environments', str(suite_environment)]
                with (output/('score_'+suite+'.log')).open('a') as log:
                    subprocess.run(command, cwd=HERE.parents[2], stdout=log, stderr=subprocess.STDOUT, check=True)
                status['completed_stages'].append('score_'+suite)
            else:
                pending_suites.append(suite)
        if pending_suites:
            update('waiting_scorer_environment', training_complete=True, validation_complete=True,
                   generation_complete=True, complete=False, unscored_suites=pending_suites,
                   official_scorers_executed=any(s.startswith('score_') for s in status['completed_stages']),
                   pending='Official evaluator environments and weight locks; no benchmark scores fabricated')
        else:
            update('complete', complete=True, training_complete=True, validation_complete=True,
                   official_scorers_executed=bool(benchmarks))
    except Exception as error:
        update('failed', error=repr(error), complete=False)
        raise
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--grounding-service', required=True)
    parser.add_argument('--ground-steps', type=int, default=1000)
    parser.add_argument('--exploratory', action='store_true')
    parser.add_argument('--approved-review')
    parser.add_argument('--environments')
    parser.add_argument('--workers', type=int, choices=[1, 4], default=1)
    parser.add_argument('--benchmarks', nargs='*', choices=['videophy2', 'vbench'], default=['videophy2', 'vbench'])
    args = parser.parse_args()
    run(load_config(args.config), args.output, args.grounding_service, args.ground_steps,
        args.exploratory, args.approved_review, args.environments, args.benchmarks, args.workers)
