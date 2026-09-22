#!/usr/bin/env python3
"""Validate human labels, report missingness bounds, export failure/control catalogs and figures."""
import argparse
import json
import math
from pathlib import Path
from hazard import interval_hazards
from common import TAXONOMY, completed, digest, read_csv, write_csv, write_json


def validate_review(review, video, blocks):
    r = dict(review)
    label = r['failure_confirmed'].strip().upper()
    if label not in ('', 'YES', 'NO', 'UNCERTAIN'):
        raise ValueError('Unknown human decision')
    r['failure_confirmed'] = label
    horizon = float(r['reviewed_until_sec'] or 0)
    if not math.isfinite(horizon) or not 0 <= horizon <= video['duration_sec'] + 1e-6:
        raise ValueError('Invalid reviewed horizon')
    r['reviewed_until_sec'] = horizon
    if label and r['human_confidence'] not in ('high', 'medium', 'low'):
        raise ValueError('Reviewed decisions need confidence')
    r['onset_lower_sec'] = r['onset_upper_sec'] = None
    r['onset_block'] = None
    if label == 'YES':
        if r['failure_type'] not in TAXONOMY or r['onset_kind'] not in ('abrupt', 'gradual_drift'):
            raise ValueError('Confirmed failure needs taxonomy and onset kind')
        if r['pre_onset_normal'] not in ('YES', 'NO', 'UNCERTAIN'):
            raise ValueError('Confirmed failure needs pre-onset history assessment')
        if r['failure_onset_block'] != '':
            b = int(r['failure_onset_block'])
            if not 0 <= b < len(blocks):
                raise ValueError('Onset block out of range')
            r['onset_block'] = b
            r['onset_lower_sec'], r['onset_upper_sec'] = blocks[b]['start_sec'], blocks[b]['end_sec']
            if r['failure_onset_frame'] != '':
                f = int(r['failure_onset_frame'])
                if not blocks[b]['frame_start'] <= f < blocks[b]['frame_end']:
                    raise ValueError('Onset frame does not belong to onset block')
                r['onset_lower_sec'] = r['onset_upper_sec'] = f / video['fps']
            if horizon < r['onset_upper_sec']:
                raise ValueError('Reviewed horizon must cover the onset frame/block')
        elif r['onset_kind'] != 'gradual_drift':
            raise ValueError('Abrupt failure requires an onset block')
        elif r['failure_onset_frame']:
            raise ValueError('Onset frame requires a corresponding block')
        else:
            if horizon <= 0:
                raise ValueError('Gradual failure needs a positive reviewed horizon')
            r['onset_lower_sec'], r['onset_upper_sec'] = 0, horizon
    elif r['failure_type'] or r['failure_onset_block'] or r['failure_onset_frame'] or r['onset_kind'] or r['pre_onset_normal']:
        raise ValueError('Failure details require a confirmed YES label')
    if label == 'NO' and horizon <= 0:
        raise ValueError('NO requires a positive reviewed horizon')
    return r


def prefix_status(review, seconds):
    if review['failure_confirmed'] == 'YES':
        if review['onset_upper_sec'] < seconds:
            return 'failure'
        # A block interval is [start,end), so an end equal to T is already observed.
        if review['onset_upper_sec'] == seconds and review['onset_upper_sec'] != review['onset_lower_sec']:
            return 'failure'
        if review['onset_lower_sec'] >= seconds and review['pre_onset_normal'] == 'YES':
            return 'normal'
    if review['failure_confirmed'] == 'NO' and review['reviewed_until_sec'] >= seconds:
        return 'normal'
    return 'unknown'


def rates(videos, reviews, prefixes):
    result = []
    for t in prefixes:
        eligible = [v for v in videos if v['duration_sec'] >= t]
        states = [prefix_status(reviews[v['video_id']], t) for v in eligible]
        n, failures, unknown = len(states), states.count('failure'), states.count('unknown')
        result.append(dict(duration_sec=t, n_eligible=n, confirmed_failures=failures, unknown=unknown,
                           lower_bound=failures/n if n else None,
                           upper_bound=(failures+unknown)/n if n else None,
                           failure_rate=failures/n if n and unknown == 0 else None))
    return result


def detector_metrics(videos, reviews, candidates):
    truth = {}
    for v in videos:
        r = reviews[v['video_id']]
        if r['failure_confirmed'] == 'YES':
            truth[v['video_id']] = True
        elif r['failure_confirmed'] == 'NO' and r['reviewed_until_sec'] >= v['duration_sec']:
            truth[v['video_id']] = False
    output = {}
    for metric in ('subject', 'attribute', 'combined'):
        predicted = {r['video_id'] for r in candidates if
                     (r['selection'] == 'candidate' if metric == 'combined' else
                      r['selected_'+metric] == 'True' or r['selected_'+metric+'_change'] == 'True')}
        tp = sum(truth.get(v) is True for v in predicted)
        fp = sum(truth.get(v) is False for v in predicted)
        fn = sum(yes and vid not in predicted for vid, yes in truth.items())
        full = len(truth) == len(videos)
        positives_complete = all(v in truth for v in predicted)
        output[metric] = dict(tp=tp, fp=fp, fn_observed=fn, n_labeled=len(truth), n_total=len(videos),
                              precision=tp/(tp+fp) if positives_complete and tp+fp else None,
                              recall=tp/(tp+fn) if full and tp+fn else None,
                              reviewed_subset_recall=tp/(tp+fn) if tp+fn else None,
                              full_cohort_reviewed=full)
    return output


def snapshot_verified(run, video, block, plan_signature):
    if block is None or block < 0:
        return False, ''
    path = run / 'outputs' / video['video_id'] / f'snapshot_after_{block:03d}.pt'
    meta = path.with_suffix('.json')
    if not path.exists() or not meta.exists():
        return False, ''
    obj = json.loads(meta.read_text())
    valid = (obj.get('verified') is True and obj['plan_signature'] == plan_signature and
             obj['after_block'] == block and obj['video_id'] == video['video_id'] and
             obj['replay_sha256'] == video['artifact_sha256']['replay.pt'] and
             obj['snapshot_sha256'] == digest(path))
    return valid, str(path.relative_to(run)) if valid else ''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args()
    run = args.run_dir.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    blocks = json.loads((run / 'block_map.json').read_text())['blocks']
    videos = completed(run)
    by_id = {v['video_id']: v for v in videos}
    raw = read_csv(run / 'human_review.csv')
    if len(raw) != len(by_id) or {r['video_id'] for r in raw} != set(by_id):
        raise ValueError('Need exactly one review row per completed video, including blank rows')
    reviews = {r['video_id']: validate_review(r, by_id[r['video_id']], blocks) for r in raw}
    candidates = read_csv(run / 'manifests/candidate_failures.csv')
    if len(candidates) != len(videos) or {c['video_id'] for c in candidates} != set(by_id):
        raise ValueError('Candidate table does not match completed cohort')
    auto = {r['video_id']: r for r in candidates}
    prefixes = [t for t in (15,30,45,60) if t <= plan['config']['duration_sec']]
    statistics = dict(planned=len(plan['videos']), completed=len(videos),
                      missing_generation=len(plan['videos'])-len(videos),
                      prefix_rates=rates(videos, reviews, prefixes),
                      interval_hazards=interval_hazards(videos, reviews, [0]+prefixes),
                      detector=detector_metrics(videos, reviews, candidates),
                      interpretation='Cumulative first-event incidence is nondecreasing by definition; it does not establish increasing per-block hazard. Missing reviews are unknown; bounds are identification bounds, not confidence intervals.')
    statistics['by_category_difficulty'] = {}
    for category, difficulty in sorted({(v['category'],v['difficulty']) for v in videos}):
        subset = [v for v in videos if (v['category'],v['difficulty']) == (category,difficulty)]
        statistics['by_category_difficulty'][category+'/'+difficulty] = rates(subset, reviews, prefixes)
    write_csv(run / 'manifests/failure_hazard.csv', statistics['interval_hazards'])
    comparison = []
    for v in videos:
        r, a = reviews[v['video_id']], auto[v['video_id']]
        b = a['suggested_onset_block']
        comparison.append(dict(video_id=v['video_id'], human_failure=r['failure_confirmed'],
                               reviewed_until_sec=r['reviewed_until_sec'],
                               auto_candidate=a['selection']=='candidate', selection=a['selection'],
                               human_onset_block=r['failure_onset_block'],
                               human_onset_lower_sec=r['onset_lower_sec'], human_onset_upper_sec=r['onset_upper_sec'],
                               auto_onset_block=b, auto_onset_sec=blocks[int(b)]['start_sec'] if b!='' else '',
                               human_failure_type=r['failure_type']))
    write_csv(run / 'manifests/human_auto_comparison.csv', comparison)
    catalog = []
    for v in videos:
        r = reviews[v['video_id']]
        if r['failure_confirmed'] != 'YES':
            continue
        onset = r['onset_block']
        pre = onset-1 if onset is not None and onset>0 else None
        verified, snapshot = snapshot_verified(run, v, pre, plan['signature'])
        catalog.append(dict(video_id=v['video_id'], prompt_id=v['prompt_id'], seed=v['seed'],
                            category=v['category'], difficulty=v['difficulty'], duration_sec=v['duration_sec'],
                            failure_confirmed='YES', failure_type=r['failure_type'],
                            failure_onset_sec=r['onset_lower_sec'] if r['onset_lower_sec']==r['onset_upper_sec'] else '',
                            onset_lower_sec=r['onset_lower_sec'], onset_upper_sec=r['onset_upper_sec'],
                            failure_onset_block=onset, pre_onset_block=pre, onset_kind=r['onset_kind'],
                            pre_onset_normal=r['pre_onset_normal'], auto_subject_drop=auto[v['video_id']]['subject_drop'],
                            auto_attribute_drop=auto[v['video_id']]['attribute_drop'], human_confidence=r['human_confidence'],
                            snapshot_verified=verified, snapshot_path=snapshot,
                            p0b_eligible=verified and pre is not None and r['pre_onset_normal']=='YES' and r['onset_kind']=='abrupt',
                            notes=r['notes']))
    fields = ['video_id','prompt_id','seed','category','difficulty','duration_sec','failure_confirmed','failure_type',
              'failure_onset_sec','onset_lower_sec','onset_upper_sec','failure_onset_block','pre_onset_block','onset_kind',
              'pre_onset_normal','auto_subject_drop','auto_attribute_drop','human_confidence','snapshot_verified',
              'snapshot_path','p0b_eligible','notes']
    write_csv(run / 'manifests/failure_catalog.csv', catalog, fields)
    controls, used = [], set()
    requested = math.ceil(len(catalog)*plan['config']['mining']['control_fraction'])
    for failure in catalog:
        if len(controls) >= requested:
            break
        b = failure['pre_onset_block']
        if b is None:
            continue
        f = by_id[failure['video_id']]
        matches = [v for v in videos if v['video_id'] not in used and reviews[v['video_id']]['failure_confirmed']=='NO'
                   and reviews[v['video_id']]['reviewed_until_sec'] >= v['duration_sec']
                   and all(v[k]==f[k] for k in ('category','difficulty','target_subject','motion_group'))]
        if not matches:
            continue
        control = sorted(matches, key=lambda v: (v['prompt_id'] != f['prompt_id'],v['video_id']))[0]
        used.add(control['video_id'])
        verified, snapshot = snapshot_verified(run, control, b, plan['signature'])
        controls.append(dict(video_id=control['video_id'], matched_failure_id=f['video_id'], block_index=b,
                             state_time_sec=blocks[b]['end_sec'], category=f['category'], difficulty=f['difficulty'],
                             target_subject=f['target_subject'], motion_group=f['motion_group'],
                             snapshot_verified=verified, snapshot_path=snapshot))
    write_csv(run / 'manifests/matched_controls.csv', controls,
              ['video_id','matched_failure_id','block_index','state_time_sec','category','difficulty',
               'target_subject','motion_group','snapshot_verified','snapshot_path'])
    statistics.update(confirmed_failures=len(catalog), p0b_eligible=sum(r['p0b_eligible'] for r in catalog),
                      controls_requested=requested, controls_matched=len(controls), controls_shortfall=requested-len(controls))
    write_json(run / 'manifests/summary.json', statistics)
    if not args.no_plots:
        from plots import render
        render(run, statistics, catalog, read_csv(run / 'manifests/block_scores.csv'))
    print(json.dumps(statistics, indent=2))


if __name__ == '__main__':
    main()
