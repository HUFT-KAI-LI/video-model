"""Scientific figures; undefined rates are labeled rather than drawn as zero."""
from collections import Counter


def render(run, summary, failures, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = run / 'figures'
    folder.mkdir(exist_ok=True)
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / name, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots()
    rows = summary['prefix_rates']
    xs = [r['duration_sec'] for r in rows]
    lo = [r['lower_bound'] for r in rows]
    hi = [r['upper_bound'] for r in rows]
    if xs and all(x is not None for x in lo):
        ax.fill_between(xs, lo, hi, alpha=.25, label='Unresolved-label bounds (not CI)')
        ax.plot(xs, lo, 'o-', label='Confirmed lower bound')
        if all(r['unknown']==0 for r in rows):
            ax.lines[0].set_label('Fully reviewed first-failure incidence')
        ax.legend()
    ax.set(xlabel='Prefix duration (s)', ylabel='P(first failure before T)', ylim=(0,1), xticks=xs,
           title=f"Completed {summary['completed']} / planned {summary['planned']} trajectories")
    save(fig, 'failure_rate_vs_duration.png')
    fig, ax = plt.subplots()
    bins = [0]+xs
    counts = [sum(f['onset_lower_sec']>=left and f['onset_upper_sec']<right for f in failures)
              for left,right in zip(bins,bins[1:])]
    ax.bar([f'{a}-{b}' for a,b in zip(bins,bins[1:])], counts)
    unresolved = len(failures)-sum(counts)
    ax.set(xlabel='Onset interval (s)', ylabel='Confirmed failures', title=f'Onsets fully inside bin; {unresolved} boundary/uncertain onsets excluded')
    save(fig, 'failure_onset_histogram.png')
    fig, ax = plt.subplots(figsize=(8,5))
    counts = Counter(f['failure_type'] for f in failures)
    ax.barh(list(counts), list(counts.values()))
    ax.set(xlabel='Confirmed failures', title='Human-confirmed failure taxonomy')
    save(fig, 'failure_type_distribution.png')
    fig, ax = plt.subplots()
    names = list(summary['detector'])
    for j, metric in enumerate(('precision', 'recall')):
        for i,name in enumerate(names):
            value = summary['detector'][name][metric]
            x = i + (j-.5)*.35
            if value is None:
                ax.text(x,.03,'N/A',ha='center',rotation=90)
            else:
                ax.bar(x,value,width=.3,color=('steelblue','darkorange')[j],label=metric if i==0 else None)
    ax.set(xticks=range(len(names)),xticklabels=names,ylim=(0,1),ylabel='Precision / recall',
           title='Blue: precision; orange: full-cohort recall; N/A = unresolved')
    save(fig, 'detector_precision_recall.png')
    examples = list(dict.fromkeys([f['video_id'] for f in failures]+[r['video_id'] for r in scores]))[:4]
    for metric,name in (('subject','subject_consistency_examples.png'),('attribute','attribute_consistency_examples.png')):
        fig, ax = plt.subplots()
        for vid in examples:
            seq = sorted([r for r in scores if r['video_id']==vid],key=lambda r:int(r['block_index']))
            ax.plot([float(r['start_sec']) for r in seq],[float(r[metric]) for r in seq],label=vid)
        if examples:
            ax.legend()
        ax.set(xlabel='Block start (s)',ylabel='DINO cosine' if metric=='subject' else 'CLIP target - max alternative',
               title='Appearance screening score' if metric=='subject' else 'Attribute screening score')
        save(fig,name)
