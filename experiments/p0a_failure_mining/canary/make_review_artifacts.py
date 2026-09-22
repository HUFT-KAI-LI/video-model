"""Per-video score plots and early-reference diagnostics, without human labels."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import read_csv,write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    run=args.run_dir
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan=json.loads((run/'plan.json').read_text())
    scores=read_csv(run/'manifests/block_scores.csv')
    candidates={r['video_id']:r for r in read_csv(run/'manifests/candidate_failures.csv')}
    diagnostics=[]
    for video in plan['videos']:
        vid=video['video_id']
        rows=[r for r in scores if r['video_id']==vid]
        if not rows:continue
        cache=np.load(run/'outputs/feature_cache'/f'{vid}.npz')
        early=cache['block_index']<plan['config']['mining']['reference_blocks']
        k=int(np.argmax(np.mean(cache['alternatives'][early],axis=0)))
        diagnostics.append(dict(video_id=vid,early_target_median=float(np.median(cache['target'][early])),
                                early_margin_median=float(np.median(cache['attribute_margin'][early])),
                                early_strongest_alternative=video['alternatives'][k],
                                selection=candidates[vid]['selection'],human_review_status='pending'))
        fig,axs=plt.subplots(2,1,figsize=(9,5),sharex=True,layout='constrained')
        for ax,metric,label in zip(axs,['subject','attribute'],['DINO whole-frame cosine','CLIP target - max alternative']):
            ax.plot([float(r['start_sec']) for r in rows],[float(r[metric]) for r in rows])
            ax.axvspan(0,2.8125,alpha=.1,color='green',label='Early reference')
            ax.set_ylabel(label);ax.grid(alpha=.2)
            b=candidates[vid]['suggested_onset_block']
            if b!='':
                t=next(float(r['start_sec']) for r in rows if int(r['block_index'])==int(b))
                ax.axvline(t,color='darkorange',ls='--',label=f'Auto proposal {t:.2f}s; not GT')
        axs[1].axhline(0,color='gray',ls=':')
        axs[0].legend(loc='best');axs[1].set(xlabel='Seconds',xticks=[0,15,30,45,60])
        fig.suptitle(f'{vid} / {candidates[vid]["selection"]}; human review pending')
        fig.savefig(run/'outputs'/vid/'scores.png',dpi=150);plt.close(fig)
    write_csv(run/'manifests/early_reference_diagnostics.csv',diagnostics)


if __name__=='__main__':
    main()
