"""Audit matched schedules and report fixed-Dev counterfactual metrics."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from restream.reality_data import write_json
from restream.reality_r1_training import TRAIN_BRANCHES
from restream.reality_selection import manifest_digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--updates',type=int,default=10)
    parser.add_argument('--root',type=Path,default=ROOT/'validation/reality_memory/r1_matched')
    args=parser.parse_args()
    runs={b:json.loads((args.root/b/f'train_{args.updates:04d}.json').read_text()) for b in TRAIN_BRANCHES}
    signatures=[]
    schedules=[]
    initial=[]
    for b,r in runs.items():
        if r['status']!='passed' or r['optimizer_steps']!=args.updates or not r['backbones_frozen']:
            raise ValueError('Incomplete/invalid matched run')
        s=copy.deepcopy(r['signature']);s.pop('branch');signatures.append(s)
        initial.append(r['initial_adapter_sha256'])
        schedules.append([{k:x[k] for k in ('batch_step','optimizer_steps','updated','index','sample_id','donor_sample_id','noise_seed','no_memory','learning_rate')}
                          for x in r['records']])
        if any(x['no_memory'] and (x['updated'] or not x['no_memory_unchanged']) for x in r['records']):
            raise ValueError('No-memory batch changed optimizer state')
    if not all(s==signatures[0] for s in signatures) or len(set(initial))!=1 or not all(s==schedules[0] for s in schedules):
        raise ValueError('Branches were not matched in provenance/initialization/order/noise/dropout/LR')
    summary={'status':'passed','optimizer_updates_per_branch':args.updates,'matched_schedule':True,
             'initial_adapter_sha256':initial[0],'batch_steps_per_branch':len(schedules[0]),
             'no_memory_batches':sum(x['no_memory'] for x in schedules[0]),
             'test_eligible_targets':signatures[0]['test_seal']['eligible_targets'],
             'branches':{},'report_sha256':{b:manifest_digest(args.root/b/f'train_{args.updates:04d}.json') for b in TRAIN_BRANCHES}}
    if args.updates>2:
        dev={b:json.loads((args.root/b/f'dev_{args.updates:04d}.json').read_text()) for b in TRAIN_BRANCHES}
        ids=[[(x['sample_id'],x['noise_seed']) for x in d['cases']] for d in dev.values()]
        if not all(i==ids[0] for i in ids):raise ValueError('Dev targets/noise differ across branches')
        base=[[r['variants']['none']['video_loss'] for r in d['cases']] for d in dev.values()]
        summary['no_memory_loss_max_cross_run_difference']=max(abs(a-b) for row in base[1:] for a,b in zip(row,base[0]))
        for b,d in dev.items():
            summary['branches'][b]={'metrics':d['metrics'],'mean_video_loss':d['mean_video_loss']}
        summary['dev_unique_targets']=next(iter(dev.values()))['unique_targets']
        summary['noise_seeds']=next(iter(dev.values()))['noise_seeds']
        summary['interpretation']='Preliminary 8-target Dev mechanism experiment. Inspect each trained checkpoint counterfactually; no untouched Test exists under the sealed rules. Do not infer quality from training loss alone.'
    else:
        for b,r in runs.items():
            second=next(x for x in r['records'] if x['updated'] and x['optimizer_steps']==2)
            summary['branches'][b]={'second_update_projector_grad_norm':sum(v*v for k,v in second['parameter_gradient_norms'].items() if k.startswith('projector.'))**.5,
                                     'parameter_change_norms':r['parameter_change_norms']}
    write_json(args.root/f'summary_{args.updates:04d}.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('branches','report_sha256')},indent=2))


if __name__=='__main__': main()
