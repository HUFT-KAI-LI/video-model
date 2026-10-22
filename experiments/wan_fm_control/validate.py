"""Held-out native FM diagnostics for A, with the same noise as C validation."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from ..wan_said.data import load_split
from ..wan_said.io_utils import read_jsonl, sha256, write_json, write_jsonl
from ..wan_said.runtime import (forward_fn, load_dit, load_trainable,
                               seed_everything, upstream)
from ..wan_said.pilot_v2.config import load_config
from ..wan_said.pilot_v2.execution import deterministic_training
from ..wan_said.pilot_v2.online import OnlineFeatures, experiment_identity


@deterministic_training
@torch.no_grad()
def validate(cfg, checkpoint, output):
    upstream()
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    identity = experiment_identity(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    rows = load_split(cfg['manifests'], 'val')
    if (state['group'] != 'A' or state['contract']['identity'] != identity or
            state['contract']['smoke'] or state['contract']['effective_loss'] != 'FM'):
        raise ValueError('Validation requires the matching non-smoke FM-only adapter')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(checkpoint_sha256=sha256(checkpoint), identity=identity,
                    validation_ids=[r['sample_id'] for r in rows], noise_seed_offset=30000,
                    test_used=False, teacher_used=False)
    lock = output/'validation_lock.json'
    if lock.exists() and json.loads(lock.read_text()) != contract:
        raise ValueError('Validation model or dataset changed')
    write_json(lock, contract)
    path = output/'per_video.jsonl'
    records = read_jsonl(path) if path.exists() else []
    if [r['sample_id'] for r in records] != [r['sample_id'] for r in rows[:len(records)]]:
        raise ValueError('Validation order/coverage changed')
    device = cfg['devices']['student']
    seed_everything(cfg['seed'])
    base = load_dit(cfg, device, lora=False).eval().requires_grad_(False)
    model = load_dit(cfg, device).eval()
    load_trainable(model, state['lora'])
    model.requires_grad_(False)
    scheduler = FlowMatchScheduler('Wan')
    scheduler.set_timesteps(1000, training=True)
    native, provider = forward_fn(), OnlineFeatures(cfg)
    started = time.monotonic()
    try:
        for index in range(len(records), len(rows)):
            row = rows[index]
            features = provider(row, ('text', 'vae'))
            provider.calls.clear()
            latent, context = features['vae']['latents'].to(device), features['text']['context'].to(device)
            losses = {}
            for name, dit in [('base', base), ('A', model)]:
                seed_everything(cfg['seed']+30000+index)
                pipe = SimpleNamespace(scheduler=scheduler, torch_dtype=torch.bfloat16,
                    device=device, dit=dit, in_iteration_models=('dit',), model_fn=native)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses[name] = FlowMatchSFTLoss(pipe, input_latents=latent,
                        context=context, use_gradient_checkpointing=False).item()
            if not all(math.isfinite(v) for v in losses.values()):
                raise FloatingPointError('Nonfinite held-out FM loss')
            records.append(dict(sample_id=row['sample_id'], split='val',
                noise_seed=cfg['seed']+30000+index, base_fm=losses['base'], A_fm=losses['A']))
            if len(records) % 10 == 0:
                write_jsonl(path, records)
                print('validation_A', len(records), '/', len(rows), flush=True)
            del features, latent, context
    finally:
        write_jsonl(path, records)
    if len(records) != len(rows) or len({r['sample_id'] for r in records}) != len(rows):
        raise ValueError('Incomplete or duplicated held-out validation')
    scores = dict(base_fm_mean=statistics.mean(r['base_fm'] for r in records),
                  A_fm_mean=statistics.mean(r['A_fm'] for r in records),
                  paired_fm_delta_mean=statistics.mean(r['A_fm']-r['base_fm'] for r in records))
    result = dict(complete=True, denominator=len(rows), valid=len(records), scores=scores,
        contract=contract, seconds=time.monotonic()-started, feature_cache_bytes=0,
        test_used=False, teacher_used=False, official_scorers_executed=False,
        interpretation='Held-out native FM diagnostics, not video quality or benchmark scores')
    write_json(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(validate(load_config(args.config), args.checkpoint, args.output)))
