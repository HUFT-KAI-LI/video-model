"""Three-arm comparisons from verified, identically paired official scores."""
import csv
import json
from pathlib import Path

from ..wan_said.io_utils import read_jsonl, sha256, write_json
from ..wan_said.pilot_v2.paired_eval.compare import paired_intervals, verified_summary
from ..wan_said.pilot_v2.subset import select

ARMS = ('base', 'A', 'C')


def matched(reports, suite):
    base = reports['base']
    for arm in ('A', 'C'):
        other = reports[arm]
        for key in ('jobs_sha256', 'subset_lock_sha256', 'weight_lock_sha256'):
            if base[key] != other[key]:
                raise ValueError(f'Unmatched {suite} {arm}: {key}')
        for key in ('base_weights', 'protocol'):
            if base['generation'][key] != other['generation'][key]:
                raise ValueError(f'Unmatched generation {suite} {arm}: {key}')
        if suite == 'vbench' and (base['subset_metadata_sha256'] != other['subset_metadata_sha256'] or
                any(reports[a]['prompt_count'] != 100 or reports[a]['samples_per_prompt'] != 5 for a in ARMS)):
            raise ValueError('VBench must preserve all identical prompt annotations and five seeds')


def metric_rows(reports, suite):
    rows = []
    for family in (('scores',) if suite == 'videophy2' else
                   ('scores', 'raw_dimensions', 'normalized_weighted_dimensions')):
        keys = set(reports['base'][family])
        if any(set(reports[arm][family]) != keys for arm in ARMS):
            raise ValueError('Models have different metric coverage')
        if suite == 'vbench' and family != 'scores' and len(keys) != 16:
            raise ValueError('All sixteen VBench dimensions are required')
        for key in sorted(keys):
            values = {a: reports[a][family][key] for a in ARMS}
            rows.append(dict(suite=suite, family=family, metric=key, **values,
                A_minus_base=values['A']-values['base'], C_minus_A=values['C']-values['A'],
                C_minus_base=values['C']-values['base']))
    return rows


def compare(sources, plans, locks, output, suites=('videophy2', 'vbench'), expected_checkpoints=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('json', 'csv', 'md'):
        (output/f'comparison.{suffix}').unlink(missing_ok=True)
    reports, provenance, metrics = {}, {}, []
    for suite in suites:
        jobs = select(read_jsonl(plans[suite]), plans[suite], locks[suite])
        reports[suite] = {a: verified_summary(sources[suite][a], a, suite, jobs, locks[suite]) for a in ARMS}
        if expected_checkpoints is not None:
            if any(reports[suite][a]['generation']['checkpoint_sha256'] != expected_checkpoints[a] for a in ARMS):
                raise ValueError('Scored adapter does not match the registered training checkpoint')
        matched(reports[suite], suite)
        metrics.extend(metric_rows(reports[suite], suite))
        provenance[suite] = {a: dict(path=str(Path(sources[suite][a]).resolve()),
                                    sha256=sha256(sources[suite][a])) for a in ARMS}
    vp = reports['videophy2']
    intervals = {name: paired_intervals(vp[left], vp[right]) for name, left, right in
                 [('A_minus_base', 'base', 'A'), ('C_minus_A', 'A', 'C'), ('C_minus_base', 'base', 'C')]}
    result = dict(complete=set(suites) == {'videophy2', 'vbench'}, completed_suites=list(suites),
        evaluation_scope='matched_100_prompt_subset', formal_benchmark_complete=False,
        publication_eligible=False, metrics=metrics, source_summaries=provenance,
        videophy2_paired_bootstrap=dict(seed=20261022, resamples=10000, intervals=intervals),
        vbench_coverage={a: reports['vbench'][a]['coverage'] for a in ARMS} if 'vbench' in reports else {},
        vbench_official_conditional_exclusions={a: reports['vbench'][a].get('official_conditional_exclusions', {})
            for a in ARMS} if 'vbench' in reports else {},
        limitations=['100-prompt pilot subsets, not full official benchmark results.',
            'A is FM-only LoRA, not full-parameter fine-tuning; C is FM+SAID with the same LoRA protocol.',
            'C minus A estimates the alignment effect under this single matched training seed, not universal significance.',
            'Official color exclusions may change effective denominators; consult per-model coverage.',
            'Caption/crop human review remains pending; the sealed 500-video dataset test split is not used.'])
    write_json(output/'comparison.json', result)
    with (output/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    lines = ['# Base / FM-Only A / FM+SAID C', '',
             'VideoPhy-2: identical 100 prompts/seeds. VBench: identical 100 prompts x 5 seeds.',
             'Scores/rates retain official scales. Rates are fractions, not percentages.', '',
             '| Suite | Family | Metric | Base | A (FM) | C (FM+SAID) | A - Base | C - A |',
             '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f'| {r["suite"]} | {r["family"]} | {r["metric"]} | {r["base"]:.6f} | '
              f'{r["A"]:.6f} | {r["C"]:.6f} | {r["A_minus_base"]:+.6f} | {r["C_minus_A"]:+.6f} |' for r in metrics]
    if 'vbench' in reports:
        lines += ['', '## Coverage', '', '| Dimension | Base Scheduled/Scored | A Scheduled/Scored | C Scheduled/Scored |',
                  '| --- | ---: | ---: | ---: |']
        for dim in sorted(result['vbench_coverage']['base']):
            values = [result['vbench_coverage'][a][dim] for a in ARMS]
            lines.append('| '+dim+' | '+' | '.join(f'{v["expected"]}/{v["scored"]}' for v in values)+' |')
    lines += ['', '## Limits', '']+['- '+s for s in result['limitations']]
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')
    return result
