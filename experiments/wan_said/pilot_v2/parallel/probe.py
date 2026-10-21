"""Real four-GPU C preflight: gradients, memory, synchronized ranks, and resumed updates."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from ...io_utils import sha256, write_json
from ..config import HERE


def run(config, grounding, output):
    output = Path(output)
    result = output/'parallel_execution.json'
    contract = dict(config_sha256=sha256(config), grounding_sha256=sha256(grounding))
    if result.exists():
        saved = json.loads(result.read_text())
        if saved['contract'] != contract or not saved['passed']:
            raise ValueError('Parallel preflight belongs to a different or failed run')
        return saved
    root = output/'ddp_smoke'
    fresh = output/'ddp_smoke_uninterrupted'

    def launch(folder, steps, resume=False, audit=False):
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                   '-m', 'experiments.wan_said.pilot_v2.parallel.train', '--config', str(config),
                   '--grounding', str(grounding), '--output', str(folder), '--smoke', '--max-steps', str(steps)]
        if resume:
            command += ['--resume', str(folder/'checkpoint.pt')]
        if audit:
            command.append('--gradient-audit')
        print(json.dumps(dict(real_gpu_preflight_command=command)), flush=True)
        subprocess.run(command, cwd=HERE.parents[2], check=True)

    checkpoint = root/'checkpoint.pt'
    step = torch.load(checkpoint, map_location='cpu', weights_only=True)['steps'] if checkpoint.exists() else 0
    if step == 0:
        launch(root, 1, audit=True)
        step = 1
    if step == 1:
        launch(root, 2, resume=True)
    if not (fresh/'checkpoint.pt').exists():
        launch(fresh, 2)
    resumed = torch.load(root/'checkpoint.pt', map_location='cpu', weights_only=True)
    uninterrupted = torch.load(fresh/'checkpoint.pt', map_location='cpu', weights_only=True)
    if (resumed['steps'] != 2 or uninterrupted['steps'] != 2 or
            resumed['contract'] != uninterrupted['contract']):
        raise ValueError('Real DDP resume preflight has mismatched protocols or incomplete updates')
    checks = {}
    for field in ('lora', 'projector'):
        left, right = resumed[field], uninterrupted[field]
        if set(left) != set(right):
            raise AssertionError('DDP resume changed parameter keys')
        checks[field] = dict(exact=all(torch.equal(left[k], right[k]) for k in left),
                            close=all(torch.allclose(left[k], right[k], rtol=1e-6, atol=1e-7) for k in left),
                            max_abs_difference=max((left[k]-right[k]).abs().max().item() for k in left))
    sequence = lambda s: [(r['sample_id'], r['noise_seed']) for r in s['records']]
    if sequence(resumed) != sequence(uninterrupted) or not all(v['close'] for v in checks.values()):
        raise AssertionError('Real DDP resume changed samples/noise or produced incompatible updates')
    report = json.loads((root/'report.json').read_text())
    if not resumed['gradient_audit'] or len({v['lora'] for v in report['synchronized_parameter_sha256']}) != 1:
        raise AssertionError('Missing real gradient audit or divergent DDP ranks')
    proof = dict(contract=contract, passed=True, real_four_gpu_training=True, resume_checks=checks,
                 optimizer_steps=2, unique_training_samples=8, local_batch=1, global_batch=4,
                 feature_cache_bytes=0, test_used=False, smoke_only=True, real_benchmark_scores=False,
                 last_rank_metrics=report['last_rank_metrics'])
    write_json(result, proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--grounding', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.grounding, args.output)), flush=True)
