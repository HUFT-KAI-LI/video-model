"""One-shot resumable preparation job, suitable for systemd-run --user."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import load_split
from .io_utils import sha256, write_json
from .openvid_subset import DEFAULT_ROOT
from .runtime import ROOT


def run(root, workers, attempts):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    state = dict(started=started, pid=os.getpid(), root=str(root), workers=workers,
                 code_commit=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip())

    def status(stage, **extra):
        state.update(stage=stage, updated=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), **extra)
        write_json(root / 'pipeline_status.json', state)
        print(json.dumps(state), flush=True)

    def command(stage, *extra):
        args = [sys.executable, '-u', '-m', 'experiments.wan_said.openvid_subset', stage,
                '--root', str(root), *extra]
        return subprocess.run(args, cwd=ROOT.parents[1], check=False).returncode

    try:
        for attempt in range(1, attempts + 1):
            status('downloading', attempt=attempt, status='running')
            result = command('download', '--workers', str(workers))
            if result == 0:
                break
            status('retry_wait', last_exit_code=result)
            if attempt == attempts:
                raise RuntimeError(f'Download failed after {attempts} complete passes; see download_result.json')
            time.sleep(30)
        output = ROOT / 'manifests_openvid'
        status('quality_check_and_split')
        if not (output / 'split_lock.json').exists():
            result = command('split', '--output', str(output), '--workers', '12')
            if result:
                raise RuntimeError('Video QC or source-disjoint split creation failed')
        counts = {name: len(load_split(output, name)) for name in ('train', 'val', 'test')}
        status('raw_data_ready', status='complete', counts=counts, manifests=str(output),
               split_lock_sha256=sha256(output / 'split_lock.json'),
               note='Data ready only. Feature caches, new Grounding validation and formal pilot are not run automatically.')
    except Exception as error:
        status('failed', status='failed', error=repr(error))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=str(DEFAULT_ROOT))
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--attempts', type=int, default=4)
    args = parser.parse_args()
    run(args.root, args.workers, args.attempts)
