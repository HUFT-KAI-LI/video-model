"""Small genuine generation checks, never a substitute for official scoring."""
import argparse
import gc
from pathlib import Path

import torch

from ..io_utils import write_json, write_jsonl
from .benchmarks import make_jobs, suites
from .config import load_config
from .inference import generate


def run(cfg, smoke_root, output):
    output, smoke_root = Path(output), Path(smoke_root)
    if output.exists():
        raise FileExistsError('Use a fresh generation smoke output')
    output.mkdir(parents=True)
    sets, _ = suites()
    jobs = [make_jobs(name, rows[:1])[0] for name, rows in sets.items()]
    write_jsonl(output/'jobs.jsonl', jobs)
    results = {}
    for group in ('base', 'A', 'B', 'C'):
        checkpoint = None if group == 'base' else smoke_root/group/'checkpoint.pt'
        records = generate(cfg, output/'jobs.jsonl', output/group, group, checkpoint,
                           smoke=True, smoke_frames=17, smoke_steps=4)
        if len(records) != 3 or any(r['pixel_std'] < 1 or r['temporal_pixel_delta'] <= 0 for r in records):
            raise AssertionError('Generation smoke is missing, blank, or static')
        results[group] = [dict(job_id=r['job']['job_id'], suite=r['job']['suite'],
                               pixel_std=r['pixel_std'], temporal_pixel_delta=r['temporal_pixel_delta'],
                               sha256=r['sha256'], seconds=r['seconds'],
                               teacher_used=r['teacher_used'], auxiliary_modules_used=r['auxiliary_modules_used'])
                          for r in records]
        gc.collect()
        torch.cuda.empty_cache()
    summary = dict(groups=results, frames=17, height=256, width=384, steps=4,
                   sampler='pinned official Wan UniPC', fps=16, cfg=6, shift=8,
                   official_scorers_executed=False, formal_efficacy_evidence=False,
                   note='One original prompt per suite per model; reduced geometry/steps, NOT formal 480P/81-frame evaluation')
    write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--smoke-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(load_config(args.config), args.smoke_root, args.output)
