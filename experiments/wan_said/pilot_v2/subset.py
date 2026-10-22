"""Explicit deterministic VideoPhy-2 pilot selection, never a reduced official suite."""
import json
import random
from pathlib import Path

from ..io_utils import read_jsonl, sha256, write_json
from .benchmarks import validate_official_plan


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
    contract = selection(jobs, sha256(full_plan), count, seed)
    path = Path(output)
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError('Frozen subset selection changed')
    write_json(path, contract)
    return contract


def select(jobs, full_plan, subset_lock):
    contract = json.loads(Path(subset_lock).read_text())
    expected = selection(jobs, sha256(full_plan), contract['denominator'], contract['selection_seed'])
    if contract != expected:
        raise ValueError('Subset identity, full plan, or deterministic selection changed')
    selected = set(contract['selected_job_ids'])
    return [job for job in jobs if job['job_id'] in selected]
