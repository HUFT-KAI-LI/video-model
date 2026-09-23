"""Wait for complete GPU outputs, then validate, evaluate and build review artifacts."""
import argparse
import collections
import json
from pathlib import Path
import subprocess
import sys
import time
from protocol import HERE, MODELS, jobs, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--poll-seconds',type=float,default=30)
    args=p.parse_args();run=args.run.resolve()
    expected={j['video_id'] for m in MODELS for j in jobs(m)}
    while True:
        done={p.parent.name for p in run.glob('outputs/*/complete.json')}
        failed=list(run.glob('outputs/*/failed.json'))
        status=dict(completed=len(done),expected=len(expected),counts=dict(collections.Counter(x.split('_')[0] for x in done)),
                    failures=[p.parent.name for p in failed],updated_unix=time.time())
        write_json(run/'status.json',status)
        print(json.dumps(status),flush=True)
        if failed: raise RuntimeError('Generation failure requires investigation; no automatic resampling')
        if not done.issubset(expected):raise ValueError('Unexpected video IDs')
        if done==expected:break
        time.sleep(args.poll_seconds)
    for script in ['evaluate.py','review.py','summarize.py']:
        subprocess.run([sys.executable,str(HERE/script),'--run',str(run)],check=True)
    write_json(run/'finished.json',dict(status='generation_and_proxy_complete_human_review_pending',videos=160,finished_unix=time.time()))


if __name__=='__main__':main()
