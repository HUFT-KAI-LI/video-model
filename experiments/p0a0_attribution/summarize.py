"""Human labels remain pending; proxy results are never substituted for ground truth."""
import argparse
import collections
import json
from pathlib import Path
from protocol import MODELS, jobs, read_csv, write_json, write_csv, digest

LABELS = {'CORRECT', 'INCORRECT', 'UNJUDGEABLE', ''}


def accuracy(labels):
    count = collections.Counter(labels)
    if set(count) - LABELS:
        raise ValueError('Invalid initial attribute label')
    n = len(labels)
    correct, incorrect = count['CORRECT'], count['INCORRECT']
    return dict(planned=n, correct=correct, incorrect=incorrect,
                unjudgeable=count['UNJUDGEABLE'], pending=count[''],
                accuracy_among_judgeable=correct/(correct+incorrect) if correct+incorrect else None,
                accuracy_lower_bound=correct/n if n else None,
                accuracy_upper_bound=(n-incorrect)/n if n else None,
                finalized=(count[''] == 0))


def paired_proxy(rows):
    by = {(r['pair_id'],r['model']): r for r in rows}
    pairs=[]
    for j in jobs('wan'):
        a,b = by.get((j['pair_id'],'wan')), by.get((j['pair_id'],'longlive'))
        if a is not None and b is not None:
            pairs.append(dict(pair_id=j['pair_id'],prompt_id=j['prompt_id'],motion_group=j['motion_group'],
                wan=int(a['initial_color_proxy_correct']),longlive=int(b['initial_color_proxy_correct'])))
    both=sum(r['wan'] and r['longlive'] for r in pairs)
    wan_only=sum(r['wan'] and not r['longlive'] for r in pairs)
    ll_only=sum(not r['wan'] and r['longlive'] for r in pairs)
    n=len(pairs)
    result=dict(n=n,both_correct=both,wan_only_correct=wan_only,longlive_only_correct=ll_only,
                both_incorrect=n-both-wan_only-ll_only,
                longlive_minus_wan=(ll_only-wan_only)/n if n else None)
    if n:
        import numpy as np
        groups=collections.defaultdict(list)
        for r in pairs: groups[r['motion_group']].append(r['longlive']-r['wan'])
        arrays=list(groups.values()); rng=np.random.default_rng(20260923)
        bootstrap=[]
        for _ in range(10000):
            selected=[v for k in rng.integers(0,len(arrays),size=len(arrays)) for v in arrays[k]]
            bootstrap.append(np.mean(selected))
        result['exploratory_motion_group_cluster_bootstrap_95ci']=np.quantile(bootstrap,[.025,.975]).tolist()
        result['clusters']=len(arrays)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--labels',type=Path)
    a=p.parse_args(); run=a.run.resolve(); out=run/'evaluation'; out.mkdir(exist_ok=True)
    rows=read_csv(out/'proxy_scores.csv') if (out/'proxy_scores.csv').exists() else []
    labels={}
    if a.labels:
        mapping=json.loads((run/'review'/'unblinding.json').read_text())
        import csv
        with a.labels.open(encoding='utf-8-sig', newline='') as source:
            human_rows = list(csv.DictReader(source))
        for r in human_rows:
            vid=mapping[r['blind_id']]
            if vid in labels: raise ValueError('Duplicate human annotation')
            if r['initial_attribute'] not in LABELS: raise ValueError('Invalid human label')
            labels[vid]=r['initial_attribute']
    summary=dict(kind='P0-A0 pipeline comparison; proxy is not ground truth', expected_videos=160,
                 completed_videos=len(rows), human={}, automatic_proxy={}, proxy_strata=[])
    for model in MODELS:
        subset=[r for r in rows if r['model']==model]
        summary['human'][model]=accuracy([labels.get(j['video_id'],'') for j in jobs(model)])
        summary['automatic_proxy'][model]=dict(n=len(subset),correct=sum(int(r['initial_color_proxy_correct']) for r in subset),
            rate=sum(int(r['initial_color_proxy_correct']) for r in subset)/len(subset) if subset else None)
        for key in ['category','difficulty','target_color']:
            for group in sorted({r[key] for r in subset}):
                sub=[r for r in subset if r[key]==group]
                summary['proxy_strata'].append(dict(model=model,stratum=key,group=group,n=len(sub),
                    correct=sum(int(r['initial_color_proxy_correct']) for r in sub)))
    summary['paired_proxy']=paired_proxy(rows)
    summary['human_paired']={'complete_judgeable_pairs':0,'both_correct':0,'wan_only_correct':0,'longlive_only_correct':0,'both_incorrect':0}
    for job in jobs('wan'):
        x=labels.get('wan_'+job['pair_id']);y=labels.get('longlive_'+job['pair_id'])
        if x not in ('CORRECT','INCORRECT') or y not in ('CORRECT','INCORRECT'): continue
        h=summary['human_paired'];h['complete_judgeable_pairs']+=1
        h['both_correct' if x==y=='CORRECT' else 'wan_only_correct' if x=='CORRECT' else 'longlive_only_correct' if y=='CORRECT' else 'both_incorrect']+=1
    write_json(out/'summary.json',summary)
    lines=['# P0-A0 results','',f"Validated videos: {len(rows)}/160.",'',
           '**Primary Initial Attribute Accuracy requires human review. CLIP below is only a whole-frame color proxy.**','',
           '| Model | Human correct / judgeable | Pending | CLIP color proxy |','|---|---:|---:|---:|']
    for m in MODELS:
        h=summary['human'][m];s=summary['automatic_proxy'][m]
        lines.append(f"| {m} | {h['correct']} / {h['correct']+h['incorrect']} | {h['pending']} | {s['correct']} / {s['n']} |")
    lines += ['', 'The same integer seeds pair prompt conditions, not identical noise tensors. Wan uses 50-step UniPC, CFG 6, shift 8 and its official negative prompt; LongLive uses its released causal 4-step model plus LoRA. A difference is a pipeline-level observation, not isolated evidence against causalization.',
              '', 'All initial mismatches are retained. Recoverability analysis must separately require initially correct → later failure.',
              '', 'Exploratory uncertainty resamples the eight shared motion/template groups. The 20 prompts are not 20 independent semantic templates. No automatic threshold was fitted to these outcomes.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
