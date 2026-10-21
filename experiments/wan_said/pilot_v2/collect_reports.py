"""Collect small review artifacts; never copy raw videos, weights, or feature tensors."""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from ..io_utils import sha256, write_json
from ..runtime import ROOT
from .config import HERE, load_config


def collect(cfg, audit, smoke, resume, inference, benchmark, output):
    audit, smoke, resume, inference, benchmark, output = map(
        Path, (audit, smoke, resume, inference, benchmark, output))
    if output.exists():
        raise FileExistsError('Keep previous review reports; choose another output directory')
    required = [audit/'audit.json', smoke/'summary.json', resume/'verification.json',
                inference/'summary.json', benchmark/'suite_lock.json']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f'Cannot claim completion; required evidence is missing: {path}')
    output.mkdir(parents=True)

    def copy(source, destination):
        destination = output/destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    copy(audit/'audit.json', 'preprocessing/audit.json')
    copy(audit/'human_review.csv', 'preprocessing/human_review_pending.csv')
    for image in audit.glob('*.jpg'):
        copy(image, Path('preprocessing/previews')/image.name)
    copy(Path(cfg['manifests'])/'split_lock.json', 'split_lock.json')
    for name in ('ground_full', 'ground_said', 'A', 'B', 'C'):
        for filename in ('report.json', 'steps.jsonl', 'gradient_audit.json'):
            if (smoke/name/filename).exists():
                copy(smoke/name/filename, Path('smoke')/name/filename)
    copy(smoke/'summary.json', 'smoke/summary.json')
    copy(resume/'verification.json', 'resume/verification.json')
    for name in ('A_resumed', 'A_uninterrupted', 'C_resumed', 'ground_said_resumed'):
        for filename in ('report.json', 'steps.jsonl'):
            copy(resume/name/filename, Path('resume')/name/filename)
    copy(inference/'summary.json', 'generation/summary.json')
    for group in ('base', 'A', 'B', 'C'):
        for filename in ('generation_lock.json', 'progress.json'):
            copy(inference/group/filename, Path('generation')/group/filename)
        for record in (inference/group/'records').glob('*.json'):
            copy(record, Path('generation')/group/'records'/record.name)
    copy(benchmark/'suite_lock.json', 'benchmarks/suite_lock.json')
    for suite in ('t2v_compbench', 'vbench', 'videophy2'):
        command = [sys.executable, '-m', 'experiments.wan_said.pilot_v2.score', '--suite', suite,
                   '--jobs', str(benchmark/f'{suite}.jsonl'), '--videos', str(inference/'C'),
                   '--output', str(ROOT/'runs/pilot_v2/scoring_plans'/suite), '--plan-only']
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        copy(ROOT/'runs/pilot_v2/scoring_plans'/suite/'score_plan.json', Path('benchmarks')/suite/'score_plan.json')
    tests = []
    for name, folder in [('shared', ROOT/'tests'), ('download', ROOT/'protocol_v2/tests'), ('pilot', HERE/'tests')]:
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(folder), '-v']
        env = dict(os.environ, WAN_SAID_REAL_VIDEO_TEST_ROOT=str(ROOT/'datasets/openvid'))
        with (output/(name+'_tests.log')).open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        tests.append(dict(name=name, command=command, returncode=result.returncode,
                          real_video_integration_enabled=True, log=name+'_tests.log'))
        if result.returncode:
            write_json(output/'failed_tests.json', tests)
            raise RuntimeError(f'Tests failed: inspect {name}_tests.log')
    packages = {name: version(name) for name in ('torch', 'transformers', 'peft', 'timm', 'av',
                                                'diffusers', 'flash-attn', 'imageio', 'numpy')}
    write_json(output/'environment.json', dict(python=sys.version, platform=platform.platform(),
               packages=packages, devices=cfg['devices'], gpu=subprocess.check_output([
                   'nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'], text=True)))
    sources = {str(path.relative_to(HERE)): sha256(path) for path in HERE.rglob('*')
               if path.is_file() and path.suffix in ('.py', '.json', '.md') and
               'reports' not in path.relative_to(HERE).parts and '__pycache__' not in path.parts}
    write_json(output/'code_snapshot_sha256.json', sources)
    journal = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                   branch=subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip(),
                   base_commit_before_this_change=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   evidence_inputs={str(p): sha256(p) for p in required}, tests=tests,
                   task_scope='Existing 5K pilot, prioritize Experiment 1 algorithm adaptation; no full formal training',
                   completed=['strict ViT-Base migration', 'full-PTS shared video crop', 'online-only frozen encoders',
                              'real 49x480x720 A/B/C training and separate gradient audits', 'resume verification',
                              'pinned native UniPC reduced-size real generation', 'official prompt plans/leakage checks'],
                   pending=['named human caption/crop/event review', 'validated non-smoke Grounding pretraining',
                            'formal A/B/C training', 'official evaluator environments and weights',
                            'real official evaluator integration and complete benchmark runs',
                            'separate early VideoPhy reproduction if a 344-prompt paper comparison is needed'],
                   feature_cache_bytes=0, full_dataset_ready=False, formal_efficacy_evidence=False,
                   raw_videos_or_weights_copied=False, official_scorers_executed=False)
    write_json(output/'operations.json', journal)
    write_json(output/'artifact_sha256.json', {str(p.relative_to(output)): sha256(p)
               for p in output.rglob('*') if p.is_file()})
    print(json.dumps(dict(report_directory=str(output.resolve()), tests=tests), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    for name in ('audit', 'smoke', 'resume', 'inference', 'benchmark', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    collect(load_config(args.config), args.audit, args.smoke, args.resume, args.inference,
            args.benchmark, args.output)
