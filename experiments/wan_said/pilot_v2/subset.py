"""Explicit deterministic VideoPhy-2 pilot selection, never a reduced official suite."""
import json
import random
from pathlib import Path

from ..io_utils import read_jsonl, sha256, write_json
from .benchmarks import check_vendor, source_hash, suites, validate_official_plan


def selection(jobs, full_plan_sha256, count=100, seed=20261022):
    if not 0 < count < len(jobs) or any(j['suite'] != 'videophy2' for j in jobs):
        raise ValueError('A pilot must be a proper, nonempty VideoPhy-2 subset')
    indices = sorted(random.Random(seed).sample(range(len(jobs)), count))
    return dict(evaluation_scope='subset_pilot', formal_benchmark_complete=False,
                full_plan_sha256=full_plan_sha256, full_benchmark_denominator=len(jobs),
                denominator=count, selection_seed=seed, selection_method='python_random_sample_sorted_indices',
                selected_job_ids=[jobs[i]['job_id'] for i in indices], scores_used_for_selection=False)


def prepare(full_plan, output, count=100, seed=20261022):
    jobs = read_jsonl(full_plan)
    validate_official_plan(jobs)
    choose = vbench_selection if jobs[0]['suite'] == 'vbench' else selection
    contract = choose(jobs, sha256(full_plan), count, seed)
    path = Path(output)
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Frozen subset selection changed')
    write_json(path, contract)
    return contract


def select(jobs, full_plan, subset_lock):
    contract = json.loads(Path(subset_lock).read_text())
    choose = vbench_selection if jobs and jobs[0]['suite'] == 'vbench' else selection
    expected = choose(jobs, sha256(full_plan), contract.get('prompt_count', contract['denominator']),
                      contract['selection_seed'])
    if contract != expected:
        raise ValueError('Subset identity, full plan, or deterministic selection changed')
    selected = set(contract['selected_job_ids'])
    return [job for job in jobs if job['job_id'] in selected]


def vbench_selection(jobs, full_plan_sha256, count=100, seed=20261022):
    sets, audit = suites()
    prompts = sets['vbench']
    dimensions = sorted(audit['vbench']['dimensions'])
    if not 16 <= count < len(prompts) or any(j['suite'] != 'vbench' for j in jobs):
        raise ValueError('VBench requires 16 or more prompt groups and all five seeds per selected prompt')
    order = list(range(len(prompts)))
    random.Random(seed).shuffle(order)
    selected, coverage = set(), dict.fromkeys(dimensions, 0)

    def add(index):
        selected.add(index)
        for dimension in prompts[index]['dimensions']:
            coverage[dimension] += 1

    # Secure five prompt groups per dimension before filling from the same shuffle.
    quota = min(5, count//len(dimensions))
    rare_first = sorted(dimensions, key=lambda d: (sum(d in p['dimensions'] for p in prompts), d))
    for dimension in rare_first:
        for index in order:
            if coverage[dimension] >= quota:
                break
            if index not in selected and dimension in prompts[index]['dimensions']:
                add(index)
    for index in order:
        if len(selected) >= count:
            break
        if index not in selected:
            add(index)
    if len(selected) != count or not all(coverage.values()):
        raise ValueError('Unable to cover all 16 dimensions within the fixed prompt count')
    ids = [p['id'] for i, p in enumerate(prompts) if i in selected]
    selected_ids = set(ids)
    selected_jobs = [j for j in jobs if j['prompt_id'] in selected_ids]
    if len(selected_jobs) != count*5:
        raise ValueError('The parent VBench plan does not contain five jobs per selected prompt')
    return dict(evaluation_scope='subset_pilot', suite='vbench', formal_benchmark_complete=False,
                full_plan_sha256=full_plan_sha256, full_benchmark_denominator=len(jobs),
                full_prompt_count=len(prompts), denominator=len(selected_jobs), prompt_count=count,
                samples_per_prompt=5, selection_seed=seed, selection_method='rarest_first_minimum_coverage_then_shuffle',
                minimum_prompt_groups_per_dimension=quota, prompt_groups_per_dimension=coverage,
                selected_prompt_ids=ids, selected_job_ids=[j['job_id'] for j in selected_jobs],
                scores_used_for_selection=False)


def vbench_metadata(jobs):
    repo = check_vendor('VBench')
    path = repo/'vbench/VBench_full_info.json'
    source_hash(repo, path)
    prompts = {j['original_prompt'] for j in jobs}
    metadata = [row for row in json.loads(path.read_text()) if row['prompt_en'] in prompts]
    if {r['prompt_en'] for r in metadata} != prompts:
        raise ValueError('VBench prompt metadata is missing; arbitrary custom prompts cannot supply all dimensions')
    return metadata
