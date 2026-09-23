"""Wait for validated full results, export evidence, and publish only the experiment review branch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from protocol import HERE, ROOT, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--push',action='store_true',help='Explicitly publish results using the existing git push configuration')
    a=p.parse_args();run=a.run.resolve()
    while True:
        failures=list(run.glob('outputs/*/failed.json'))
        if failures:raise RuntimeError('Generation failed: '+', '.join(str(x) for x in failures))
        summary=run/'evaluation/summary.json'
        if summary.exists() and json.loads(summary.read_text())['completed_videos']==160 and (run/'review/index.html').exists():break
        time.sleep(30)
    subprocess.run([sys.executable,str(HERE/'export_evidence.py'),'--run',str(run)],check=True,cwd=ROOT)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py'],check=True,cwd=ROOT)
    subprocess.run(['node',str(HERE/'test_review.cjs'),str(run/'review/index.html')],check=True,cwd=ROOT)
    if a.push:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        if branch!='codex/p0a-natural-failure-mining':raise RuntimeError('Unexpected branch; refuse publication')
        # This isolated checkout is dedicated to the experiment. Never include unrelated staged files.
        if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip():
            raise RuntimeError('Existing staged changes require review; refuse publication')
        subprocess.run(['git','add','experiments/p0a0_attribution/evidence','experiments/p0a0_attribution/RESULTS.md'],check=True,cwd=ROOT)
        subprocess.run(['git','diff','--cached','--check'],check=True,cwd=ROOT)
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','-m',
                        'Report complete 160-video P0-A0 comparison with validated automatic proxy evidence'],check=True,cwd=ROOT)
        subprocess.run(['git','push','origin',branch],check=True,cwd=ROOT,timeout=120)
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(run/'published.json',dict(branch=branch,commit=commit,human_accuracy='pending'))
    print('FINISHED',flush=True)


if __name__=='__main__':main()
