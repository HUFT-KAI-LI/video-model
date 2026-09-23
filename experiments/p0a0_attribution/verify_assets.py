"""Recheck local model identities against the previously pinned official Canary assets."""
import argparse
import json
from pathlib import Path
from protocol import ROOT, digest, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    expected=json.loads((ROOT/'experiments/p0a_failure_mining/canary/evidence/provenance.json').read_text())['weights']
    checked={}
    for key,value in expected.items():
        if key.startswith('wan/'):
            path=ROOT/'restream_mvp/models/Wan2.1-T2V-1.3B'/key[4:]
        else:
            paths=list((ROOT/'restream_mvp/models/LongLive-1.3B').rglob(key))
            if len(paths)!=1:raise ValueError(key)
            path=paths[0]
        actual=digest(path)
        if actual!=value:raise ValueError(f'Model identity mismatch: {key}')
        checked[key]=actual
        print('VERIFIED',key,flush=True)
    write_json(a.run/'weights_verified.json',dict(weights=checked,reference='experiments/p0a_failure_mining/canary/evidence/provenance.json'))


if __name__=='__main__':main()
