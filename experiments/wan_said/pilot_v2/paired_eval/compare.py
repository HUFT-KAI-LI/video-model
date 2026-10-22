"""Validate four complete official result sets before producing paired comparisons."""
import csv
import json
from pathlib import Path

import numpy as np

from ...io_utils import read_jsonl, sha256, write_json
from ..evaluate_c import recorded_jobs
from ..subset import select


def verified_summary(path, arm, suite, jobs, subset_lock):
    path = Path(path)
    report = json.loads(path.read_text())
    status = json.loads((path.parent/'score_status.json').read_text())
    if (status.get('summary_sha256') != sha256(path) or not status['official_scorers_executed'] or
            not status['subset_table_ready'] or any(t.get('returncode') != 0 for t in status['tasks'])):
        raise ValueError('Summary is not backed by completed official scorer execution and its checksum')
    if (report['suite'] != suite or report['model'] != arm or not report['complete'] or
            not report['subset_table_ready'] or report['main_table_ready'] or
            report['formal_benchmark_complete'] or report['denominator'] != len(jobs) or
            report['subset_lock_sha256'] != sha256(subset_lock)):
        raise ValueError('Missing, incomplete, relabeled or mismatched paired subset result')
    videos = path.parent.parent/'videos'
    generation_path = videos/'generation_lock.json'
    generation = json.loads(generation_path.read_text())
    if (sha256(generation_path) != report['generation_lock_sha256'] or
            generation != report['generation'] or generation['group'] != arm or
            (generation['checkpoint_sha256'] is None) != (arm == 'base')):
        raise ValueError('Generation/model identity changed after scoring')
    if len(recorded_jobs(jobs, videos, verify=True)) != len(jobs):
        raise ValueError('Not every matched generation has a verified record')
    for name, expected in report['result_sha256'].items():
        raw = path.parent/f'videophy2_{name}.csv' if suite == 'videophy2' else Path(name)
        if sha256(raw) != expected:
            raise ValueError('Official raw scores changed after aggregation')
    if suite == 'vbench' and (len(report['raw_dimensions']) != 16 or len(report['coverage']) != 16):
        raise ValueError('All sixteen VBench dimensions must be present')
    return report


def paired_intervals(base, adapted, seed=20261022, resamples=10000):
    keys = ('sa_mean', 'pc_mean', 'sa_at_4', 'pc_at_4', 'joint_at_4')
    b = {r['job_id']: r for r in base['paired_results']}
    c = {r['job_id']: r for r in adapted['paired_results']}
    if (len(base['paired_results']) != 100 or len(adapted['paired_results']) != 100 or
            len(b) != 100 or set(b) != set(c)):
        raise ValueError('The same 100 VideoPhy-2 jobs must be paired')

    def values(row):
        return [row['sa'], row['pc'], row['sa'] >= 4, row['pc'] >= 4,
                row['sa'] >= 4 and row['pc'] >= 4]

    delta = np.array([values(c[k]) for k in sorted(b)], dtype=float)-np.array(
        [values(b[k]) for k in sorted(b)], dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(b), size=(resamples, len(b)))
    samples = delta[indices].mean(axis=1)
    bounds = np.quantile(samples, [.025, .975], axis=0)
    return {key: dict(delta=float(delta[:, i].mean()), ci95=[float(v) for v in bounds[:, i]],
                     c_better=int((delta[:, i] > 0).sum()), tied=int((delta[:, i] == 0).sum()),
                     c_worse=int((delta[:, i] < 0).sum())) for i, key in enumerate(keys)}


def compare(results, plans, subset_locks, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Stale final tables must never survive a failed revalidation.
    for name in ('comparison.csv', 'comparison.md', 'comparison.json'):
        (output/name).unlink(missing_ok=True)
    reports, provenance = {}, {}
    for suite in ('videophy2', 'vbench'):
        jobs = select(read_jsonl(plans[suite]), plans[suite], subset_locks[suite])
        reports[suite] = {arm: verified_summary(results[suite][arm], arm, suite, jobs, subset_locks[suite])
                          for arm in ('base', 'C')}
        base, adapted = reports[suite]['base'], reports[suite]['C']
        for key in ('jobs_sha256', 'subset_lock_sha256', 'weight_lock_sha256'):
            if base[key] != adapted[key]:
                raise ValueError(f'Unmatched paired evaluation identity: {suite} {key}')
        for key in ('base_weights', 'protocol'):
            if base['generation'][key] != adapted['generation'][key]:
                raise ValueError(f'Unmatched generation protocol/weights: {suite} {key}')
        if suite == 'vbench' and (base['subset_metadata_sha256'] != adapted['subset_metadata_sha256'] or
                base['prompt_count'] != 100 or adapted['prompt_count'] != 100 or
                base['samples_per_prompt'] != 5 or adapted['samples_per_prompt'] != 5):
            raise ValueError('VBench must use the same 100 annotated prompts and five seeds')
        provenance[suite] = {arm: dict(path=str(Path(results[suite][arm]).resolve()),
                                     sha256=sha256(results[suite][arm])) for arm in ('base', 'C')}
    intervals = paired_intervals(reports['videophy2']['base'], reports['videophy2']['C'])
    rows = []

    def add(suite, family, key, base, adapted):
        row = dict(suite=suite, family=family, metric=key, base=base, C=adapted, delta_C_minus_base=adapted-base)
        if suite == 'videophy2':
            row.update(delta_ci95_low=intervals[key]['ci95'][0], delta_ci95_high=intervals[key]['ci95'][1])
        rows.append(row)

    for suite, arms in reports.items():
        for key in arms['base']['scores']:
            add(suite, 'aggregate', key, arms['base']['scores'][key], arms['C']['scores'][key])
        if suite == 'vbench':
            for field in ('raw_dimensions', 'normalized_weighted_dimensions'):
                for key in sorted(arms['base'][field]):
                    add(suite, field, key, arms['base'][field][key], arms['C'][field][key])
    report = dict(complete=True, paired_comparison_ready=True, evaluation_scope='matched_100_prompt_subset',
                  formal_benchmark_complete=False, publication_eligible=False,
                  prompt_counts=dict(videophy2=100, vbench=100),
                  videos_per_model=dict(videophy2=100, vbench=500), metrics=rows,
                  videophy2_paired_bootstrap=dict(seed=20261022, resamples=10000, intervals=intervals),
                  vbench_coverage={arm: reports['vbench'][arm]['coverage'] for arm in ('base', 'C')},
                  vbench_official_conditional_exclusions={arm: reports['vbench'][arm].get(
                      'official_conditional_exclusions', {}) for arm in ('base', 'C')},
                  source_summaries=provenance,
                  limitations=['Subset results, not full official benchmark scores.',
                      'Official color may exclude undetected targets; scheduled and scored counts are reported separately.',
                      'Base versus C measures combined fine-tuning effects; it does not isolate SAID from FM-only fine-tuning.',
                      'The C adapter remains exploratory; human review and the sealed dataset test split are not completed.'])
    write_json(output/'comparison.json', report)
    fields = ['suite', 'family', 'metric', 'base', 'C', 'delta_C_minus_base', 'delta_ci95_low', 'delta_ci95_high']
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Matched Base Versus C: 100-Prompt Subsets', '',
             'VideoPhy-2: 100 videos/model. VBench: 100 prompts x 5 seeds = 500 videos/model.',
             'Same prompt IDs, seeds, base weights, native generation protocol and scoring environment per pair.',
             'SA/PC means use 1..5; threshold rates and official aggregate scores are fractions, not percentages.',
             'VideoPhy-2 paired bootstrap 95% intervals are recorded in comparison.json and comparison.csv.', '',
             '| Suite | Family | Metric | Base | C | C - Base |', '| --- | --- | --- | ---: | ---: | ---: |']
    lines.extend(f'| {r["suite"]} | {r["family"]} | {r["metric"]} | {r["base"]:.6f} | '
                 f'{r["C"]:.6f} | {r["delta_C_minus_base"]:+.6f} |' for r in rows)
    lines.extend(['', '## Coverage', '', '| Dimension | Base Scheduled / Scored | C Scheduled / Scored |',
                  '| --- | ---: | ---: |'])
    for dimension in sorted(report['vbench_coverage']['base']):
        b, c = [report['vbench_coverage'][arm][dimension] for arm in ('base', 'C')]
        lines.append(f'| {dimension} | {b["expected"]} / {b["scored"]} | {c["expected"]} / {c["scored"]} |')
    lines.extend(['', '## Limits', '']+['- '+item for item in report['limitations']])
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return report
