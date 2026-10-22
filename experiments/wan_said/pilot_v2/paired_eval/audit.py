"""Archive small evaluation evidence and optionally publish only that directory."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ...io_utils import sha256, write_json
from ...runtime import ROOT
from ..config import HERE

REPO = HERE.parents[2]


def snapshot(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Audit snapshots are immutable; use a fresh directory')
    output.mkdir(parents=True)
    for source in run.rglob('*'):
        if not source.is_file() or source.suffix not in ('.json', '.jsonl', '.csv', '.log', '.md'):
            continue
        if source.stat().st_size > 20*2**20:
            raise ValueError(f'Unexpected large evidence artifact: {source}')
        target = output/'run'/source.relative_to(run)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    tests = []
    for name, folder in [('shared', ROOT/'tests'), ('download', ROOT/'protocol_v2/tests'), ('pilot', HERE/'tests')]:
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', str(folder), '-v']
        with (output/f'{name}_tests.log').open('w') as log:
            result = subprocess.run(command, cwd=REPO,
                env=dict(os.environ, WAN_SAID_REAL_VIDEO_TEST_ROOT=str(ROOT/'datasets/openvid')),
                stdout=log, stderr=subprocess.STDOUT)
        tests.append(dict(name=name, returncode=result.returncode))
    services = {}
    for unit in ('wan-said-paired100-20261022', 'wan-said-vbench-prepare-paired100-20261022'):
        result = subprocess.run(['systemctl', '--user', 'show', unit, '--property=ActiveState',
                                 '--property=SubState', '--property=MainPID', '--property=Result'],
                                capture_output=True, text=True)
        services[unit] = result.stdout
    write_json(output/'snapshot.json', dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        tests=tests, services=services, checkpoints_or_raw_videos_uploaded=False, feature_cache_enabled=False))
    write_json(output/'source_sha256.json', {str(p.relative_to(HERE)): sha256(p) for p in HERE.rglob('*.py')
        if 'reports' not in p.parts and '__pycache__' not in p.parts})
    write_json(output/'artifact_sha256.json', {str(p.relative_to(output)): sha256(p)
        for p in output.rglob('*') if p.is_file()})
    if any(t['returncode'] for t in tests):
        raise RuntimeError('Audit tests failed; inspect preserved logs')
    return output


def publish(output):
    output = Path(output).resolve()
    if not output.is_relative_to(HERE/'reports'):
        raise ValueError('Only the experiment audit directory may be published')
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=REPO).returncode:
        raise ValueError('Refusing to commit unrelated staged changes')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    if branch != 'codex/wan-said-c-eval':
        raise ValueError('Unexpected branch for audit publication')
    relative = str(output.relative_to(REPO))
    subprocess.run(['git', 'add', '--', relative], cwd=REPO, check=True)
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=REPO).split(b'\0')
    if any(p and not p.decode().startswith(relative+'/') for p in staged):
        raise ValueError('Concurrent unrelated staging detected; no commit was made')
    subprocess.run(['git', '-c', 'user.name=Codex', '-c', 'user.email=codex@localhost', 'commit',
                    '-m', 'Archive matched base/C 100-prompt evaluation results and audit'], cwd=REPO, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    environment = dict(os.environ, GIT_SSH_COMMAND='ssh -p 443 -o BatchMode=yes -o ConnectTimeout=20 '
                       '-o IdentitiesOnly=yes -i /home/likai/.ssh/id_ed25519_video_model')
    subprocess.run(['git', '-c', 'remote.origin.pushurl=ssh://git@ssh.github.com:443/HUFT-KAI-LI/video-model.git',
                    'push', 'origin', branch], cwd=REPO, env=environment, check=True, timeout=120)
    return commit


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    directory = snapshot(args.run, args.output)
    print(json.dumps(dict(output=str(directory), commit=publish(directory) if args.publish else None)))
