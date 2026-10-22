"""Snapshot C progress and tests for Git review without uploading data or checkpoints."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..io_utils import sha256, write_json
from ..runtime import ROOT
from .config import HERE


def collect(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Keep earlier progress snapshots; choose a fresh report directory')
    output.mkdir(parents=True)
    copied = []
    for pattern in ('run_lock.json', 'run_lock.*.json', 'status.json', 'subset_lock.json', 'reuse_report.json',
                    'budget_decision.json', 'resolved_config.json', '*.log',
                    'grounding/report.json', 'grounding/steps.jsonl', 'C/report.json', 'C/steps.jsonl',
                    'C/gradient_audit.json', 'validation/*.json', 'validation/*.jsonl', 'plans/suite_lock.json',
                    'scores/*/summary.*', 'scores/*/score_status.json', 'scorers/download_plan.json',
                    'scorers/*environment.json', 'scorers/download.log', 'base_timing/generation_lock.json',
                    'base_timing/progress.json', 'base_timing/records/*.json',
                    'ddp_smoke/report.json', 'ddp_smoke/steps.jsonl', 'ddp_smoke/gradient_audit.json',
                    'ddp_smoke_uninterrupted/report.json', 'parallel_execution.json',
                    'videophy2/videos/generation_lock.json', 'videophy2/videos/progress*.json',
                    'videophy2/videos/worker_*.log', 'videophy2/videos/records/*.json',
                    'vbench/videos/generation_lock.json', 'vbench/videos/progress*.json',
                    'vbench/videos/worker_*.log', 'vbench/videos/records/*.json',
                    '*/score_driver.log', '*/scores/summary.*', '*/scores/score_status.json',
                    '*/scores/videophy2_*.csv', '*/scores/*.log', '*/scores/vbench/*_eval_results.json',
                    'scorers/vbench_pip_freeze.txt', 'scorers/dependency_constraints.txt',
                    'scorer_probe*/*.json', 'scorer_probe*/*.csv', 'scorer_probe*/*.log',
                    'scorers/videophy2_pip_freeze.txt', 'scorers/*_source.json'):
        for source in run.glob(pattern):
            if source.stat().st_size > 20*2**20:
                raise ValueError(f'Large evidence file needs explicit handling: {source}')
            relative = source.relative_to(run)
            target = output/'run'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(relative))
    tests = []
    for name, folder in [('shared', ROOT/'tests'), ('download', ROOT/'protocol_v2/tests'), ('pilot', HERE/'tests')]:
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(folder), '-v']
        environment = dict(os.environ, WAN_SAID_REAL_VIDEO_TEST_ROOT=str(ROOT/'datasets/openvid'))
        with (output/(name+'_tests.log')).open('w') as log:
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        tests.append(dict(name=name, returncode=process.returncode, command=command))
    services = {}
    for name in ('wan-said-c-grounding-20261021', 'wan-said-c-pipeline-20261021', 'wan-said-c-trainval-20261021',
                 'wan-said-c-grounding-4gpu-20261021', 'wan-said-c-trainval-4gpu-20261021',
                 'wan-said-videophy2-weights-direct-20261021', 'wan-said-c-official-20261022',
                 'wan-said-vbench-setup-20261022', 'wan-said-c-subset100-20261022'):
        process = subprocess.run(['systemctl', '--user', 'show', name, '--property=ActiveState',
                                  '--property=SubState', '--property=MainPID', '--property=Result',
                                  '--property=ExecMainStartTimestamp', '--property=NRestarts'],
                                 capture_output=True, text=True)
        services[name] = process.stdout.strip().splitlines()
    write_json(output/'source_sha256.json', {str(p.relative_to(HERE)): sha256(p) for p in HERE.rglob('*.py')
               if 'reports' not in p.parts and '__pycache__' not in p.parts})
    write_json(output/'snapshot.json', dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
               branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
               parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               copied=copied, tests=tests, services=services,
               official_scores_available=any((run/p).exists() for p in
                   ('scores/videophy2/summary.csv', 'scores/vbench/summary.csv',
                    'videophy2/scores/summary.csv', 'vbench/scores/summary.csv')),
               checkpoint_or_raw_video_uploaded=False, feature_cache_bytes=0,
               note='Point-in-time progress snapshot, not a completion claim; synthetic scoring tests are not real model scores'))
    write_json(output/'artifact_sha256.json', {str(p.relative_to(output)): sha256(p)
               for p in output.rglob('*') if p.is_file()})
    if any(t['returncode'] for t in tests):
        raise RuntimeError('Tests failed; inspect preserved test logs')
    print(json.dumps(dict(output=str(output), tests=tests), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    collect(args.run, args.output)
