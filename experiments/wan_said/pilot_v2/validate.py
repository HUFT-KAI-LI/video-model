"""Held-out paired native FM diagnostics. These are not video benchmark scores."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from ..alignment import Grounding, Projector, distillation_loss, masked_pool
from ..data import load_split
from ..io_utils import read_jsonl, sha256, write_json, write_jsonl
from ..runtime import forward_fn, load_dit, load_trainable, seed_everything, upstream
from ..train_grounding import diagnostics
from .config import load_config
from .execution import deterministic_training
from .online import OnlineFeatures, experiment_identity


def paired_summary(records, expected):
    if len(records) != expected or len({r['sample_id'] for r in records}) != expected:
        raise ValueError('Validation coverage is incomplete or duplicated; denominator cannot be reduced')
    for row in records:
        if not all(math.isfinite(row[k]) for k in ('base_fm', 'c_fm', 'c_said')):
            raise ValueError('Nonfinite validation metric')
    differences = [r['c_fm']-r['base_fm'] for r in records]
    return dict(base_fm_mean=statistics.mean(r['base_fm'] for r in records),
                c_fm_mean=statistics.mean(r['c_fm'] for r in records),
                c_said_mean=statistics.mean(r['c_said'] for r in records),
                paired_fm_delta_mean=statistics.mean(differences),
                paired_fm_delta_standard_error=statistics.stdev(differences)/math.sqrt(expected) if expected > 1 else None,
                interpretation='Fixed held-out noise/timesteps, native weighted FM; not VBench/VideoPhy-2 or generation quality')


@deterministic_training
@torch.no_grad()
def validate(cfg, checkpoint, grounding, output):
    upstream()
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    identity = experiment_identity(cfg)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    ground_state = torch.load(grounding, map_location='cpu', weights_only=True)
    if (state['group'] != 'C' or state['contract']['identity'] != identity or
            state['contract']['grounding_sha256'] != sha256(grounding) or
            ground_state['contract']['identity'] != identity or ground_state['mode'] != 'said'):
        raise ValueError('C validation requires matching C/Grounding/preprocessing/split checkpoints')
    if state['contract']['smoke'] or not ground_state['grounding_validated']:
        raise ValueError('Smoke or unvalidated Grounding cannot be used as a trained C validation')
    rows = load_split(cfg['manifests'], 'val')
    contract = dict(checkpoint_sha256=sha256(checkpoint), grounding_sha256=sha256(grounding),
                    identity=identity, validation_ids=[r['sample_id'] for r in rows],
                    noise_seed_offset=30000, test_used=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'validation_lock.json'
    if lock.exists() and json.loads(lock.read_text()) != contract:
        raise ValueError('Validation output belongs to another model/protocol')
    write_json(lock, contract)
    records_path = output/'per_video.jsonl'
    records = read_jsonl(records_path) if records_path.exists() else []
    if [r['sample_id'] for r in records] != [r['sample_id'] for r in rows[:len(records)]]:
        raise ValueError('Existing validation records do not match the frozen validation order')
    device = cfg['devices']['student']
    seed_everything(cfg['seed'])
    base = load_dit(cfg, device, lora=False).eval().requires_grad_(False)
    model = load_dit(cfg, device).eval()
    load_trainable(model, state['lora'])
    model.requires_grad_(False)
    head = Projector(1536, cfg['dim']).to(device).eval().requires_grad_(False)
    head.load_state_dict(state['projector'], strict=True)
    ground = Grounding(video_dim=768, dim=cfg['dim']).to(device).eval().requires_grad_(False)
    ground.load_state_dict(ground_state['model'], strict=True)
    scheduler = FlowMatchScheduler('Wan')
    scheduler.set_timesteps(1000, training=True)
    native = forward_fn()
    captured = {}

    def capture(module, inputs, hidden):
        captured['h15'] = masked_pool(hidden).detach()

    hook = model.blocks[14].register_forward_hook(capture)
    provider, texts, videos = OnlineFeatures(cfg), [], []
    started = time.monotonic()
    try:
        for index, row in enumerate(rows):
            begin = time.monotonic()
            done = index < len(records)
            features = provider(row, ('text', 'teacher') if done else ('text', 'teacher', 'vae'))
            texts.append(features['text']['pooled'])
            videos.append(features['teacher']['pooled'])
            if done:
                continue
            latent, context = features['vae']['latents'].to(device), features['text']['context'].to(device)
            losses = {}
            for name, dit in [('base', base), ('c', model)]:
                seed_everything(cfg['seed']+30000+index)
                pipe = SimpleNamespace(scheduler=scheduler, torch_dtype=torch.bfloat16, device=device,
                                       dit=dit, in_iteration_models=('dit',), model_fn=native)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    losses[name] = FlowMatchSFTLoss(pipe, input_latents=latent, context=context,
                                                  use_gradient_checkpointing=False).item()
            target = ground.target(features['text']['pooled'].to(device), features['teacher']['pooled'].to(device))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                said = distillation_loss(head(captured.pop('h15')), target).item()
            if not all(math.isfinite(x) for x in [*losses.values(), said]):
                raise FloatingPointError('Nonfinite held-out validation metric')
            records.append(dict(sample_id=row['sample_id'], split='val', noise_seed=cfg['seed']+30000+index,
                                base_fm=losses['base'], c_fm=losses['c'], c_said=said,
                                seconds=time.monotonic()-begin))
            print('validation', records[-1], flush=True)
            if len(records) % 10 == 0:
                write_jsonl(records_path, records)
                write_json(output/'progress.json', dict(expected=len(rows), valid=len(records),
                           complete=False, denominator=len(rows), test_used=False))
    finally:
        hook.remove()
        write_jsonl(records_path, records)
    seed_everything(cfg['seed']+40000)
    retrieval = diagnostics(ground, torch.cat(texts).to(device), torch.cat(videos).to(device))
    report = dict(denominator=len(rows), valid=len(records), complete=True, test_used=False,
                  feature_cache_bytes=0, seconds=time.monotonic()-started, contract=contract,
                  scores=paired_summary(records, len(rows)), grounding_validation=retrieval,
                  retrieval_chance_r1=1/len(rows), exploratory=bool(state['contract'].get('exploratory')),
                  publication_eligible=not state['contract'].get('exploratory', False),
                  official_scorers_executed=False, formal_efficacy_evidence=False)
    write_json(output/'summary.json', report)
    write_json(output/'progress.json', dict(expected=len(rows), valid=len(records), complete=True,
               denominator=len(rows), test_used=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--grounding', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(validate(load_config(args.config), args.checkpoint, args.grounding, args.output), indent=2))
