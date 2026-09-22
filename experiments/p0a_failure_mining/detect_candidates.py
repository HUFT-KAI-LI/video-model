#!/usr/bin/env python3
"""Sustained relative drops, local changes, pool ranking and seeded negative audits."""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from common import REVIEW_FIELDS, completed, digest, read_csv, write_csv, write_json


def sustained(values, persistence, start):
    candidates = [(min(values[b:b+persistence]), b) for b in range(start, len(values)-persistence+1)]
    if not candidates:
        return 0.0, None
    score, onset = max(candidates, key=lambda item: item[0])
    return max(0.0, score), onset if score > 0 else None


def score_video(rows, config):
    rows = sorted(rows, key=lambda r: int(r['block_index']))
    if [int(r['block_index']) for r in rows] != list(range(len(rows))):
        raise ValueError('Missing/duplicate block scores')
    k, n, window = (config[x] for x in ('reference_blocks', 'persistence_blocks', 'change_window'))
    if len(rows) < k+n:
        raise ValueError('Not enough blocks for reference and persistence windows')
    result = {}
    for name in ('subject', 'attribute'):
        values = [float(r[name]) for r in rows]
        if not all(math.isfinite(v) for v in values):
            raise ValueError('Nonfinite feature scores')
        baseline = median(values[:k])
        drops = [baseline-v for v in values]
        change = [0.0 if b < k else median(values[max(0,b-window):b])-values[b] for b in range(len(values))]
        for metric, series in ((name, drops), (name+'_change', change)):
            value, onset = sustained(series, n, k)
            result[metric+'_drop'], result[metric+'_onset'] = value, onset
    return result


def select_pool(summaries, fraction):
    selected = set()
    metrics = ('subject', 'attribute', 'subject_change', 'attribute_change')
    for metric in metrics:
        ranked = sorted(summaries, key=lambda r: (-r[metric+'_drop'], r['video_id']))
        count = max(1, math.ceil(len(ranked)*fraction))
        if not ranked:
            continue
        threshold = ranked[min(count, len(ranked))-1][metric+'_drop']
        # Include ties; constant curves are not candidates even at the top percentile.
        for row in ranked:
            row['selected_'+metric] = row[metric+'_drop'] > 0 and row[metric+'_drop'] >= threshold
            if row['selected_'+metric]:
                selected.add(row['video_id'])
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    run = args.run_dir.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    cfg = plan['config']['mining']
    scores_path = run / 'manifests/block_scores.csv'
    proof = json.loads(scores_path.with_suffix('.provenance.json').read_text())
    if digest(scores_path) != proof['block_scores_sha256']:
        raise ValueError('Block scores changed after extraction')
    grouped = defaultdict(list)
    for row in read_csv(scores_path):
        grouped[row['video_id']].append(row)
    videos = {r['video_id']: r for r in completed(run)}
    if set(grouped) != set(videos):
        raise ValueError('Extract features for every completed video before ranking')
    summaries = [dict(video_id=vid, **score_video(rows, cfg)) for vid, rows in sorted(grouped.items())]
    selected = select_pool(summaries, cfg['top_fraction'])
    negatives = sorted(set(grouped)-selected)
    audit = set(random.Random(cfg['audit_seed']).sample(negatives, math.ceil(len(negatives)*cfg['audit_fraction'])))
    reviews = []
    for row in summaries:
        vid = row['video_id']
        row['selection'] = 'candidate' if vid in selected else ('audit' if vid in audit else 'unselected')
        proposals = [row[m+'_onset'] for m in ('subject','attribute','subject_change','attribute_change') if row['selected_'+m]]
        row['suggested_onset_block'] = min(proposals) if proposals else ''
        # Deliberately leave human decisions/onset blank, including candidate videos.
        review = dict.fromkeys(REVIEW_FIELDS, '')
        review.update(video_id=vid, selection=row['selection'])
        reviews.append(review)
    review_path = run / 'human_review.csv'
    if not review_path.exists():
        write_csv(review_path, reviews, REVIEW_FIELDS)
    else:
        existing = read_csv(review_path)
        if [(r['video_id'], r['selection']) for r in existing] != [(r['video_id'], r['selection']) for r in reviews]:
            raise ValueError('Pool changed after review started. Preserve old annotations and use a separate review run.')
    write_csv(run / 'manifests/candidate_failures.csv', summaries)
    write_json(run / 'manifests/mining_protocol.json', dict(config=cfg, completed=len(videos), planned=len(plan['videos']),
               candidates=len(selected), audits=len(audit), negative_audit_probability=cfg['audit_fraction'],
               realized_audit_fraction=len(audit)/len(negatives) if negatives else 0,
               scores_sha256=digest(scores_path)))
    from review_page import build
    build(run, plan, summaries)
    print(f'{len(selected)} candidates + {len(audit)} randomly selected negatives; {len(videos)} completed videos')


if __name__ == '__main__':
    main()
