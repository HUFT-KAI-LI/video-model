import argparse
import random
import time
from pathlib import Path

import torch

from ..alignment import Grounding
from ..data import load_split
from ..io_utils import write_json, write_jsonl
from ..runtime import save_torch, seed_everything
from ..train_grounding import diagnostics
from .config import load_config
from .execution import deterministic_training
from .online import OnlineFeatures, experiment_identity


def encode_batch(provider, rows, device):
    if hasattr(provider, 'encode_batch'):
        return provider.encode_batch(rows, device)
    features = [provider(row, ('text', 'teacher')) for row in rows]
    return (torch.cat([r['text']['pooled'] for r in features]).to(device),
            torch.cat([r['teacher']['pooled'] for r in features]).to(device))


@deterministic_training
def train(cfg, mode, output, steps, train_limit=None, val_limit=16, resume=None, provider=None):
    if mode not in ('full', 'said') or steps < 1:
        raise ValueError('Invalid Grounding mode or step count')
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output or an explicit resume checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg['seed'])
    identity = experiment_identity(cfg)
    rows = load_split(cfg['manifests'], 'train', training=True)
    val_rows = load_split(cfg['manifests'], 'val')
    if train_limit is not None:
        if train_limit < 2:
            raise ValueError('At least two real training negatives are required')
        rows = rows[:train_limit]
    if val_limit < 2:
        raise ValueError('Validation retrieval requires at least two heldout videos')
    val_rows = val_rows[:val_limit]
    device = cfg['devices']['grounding']
    provider = provider or OnlineFeatures(cfg)
    model = Grounding(video_dim=768, dim=cfg['dim']).to(device)
    if mode == 'full':
        model.gate.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['ground_lr'], weight_decay=cfg['weight_decay'])
    start_step, records, seen = 0, [], set()
    contract = dict(identity=identity, train_ids=[r['sample_id'] for r in rows],
                    val_ids=[r['sample_id'] for r in val_rows], mode=mode)
    old = None
    if resume:
        old = torch.load(resume, map_location='cpu', weights_only=True)
        if old['contract'] != contract:
            raise ValueError('Grounding resume changed teacher, split, data scope, mode, or optimizer protocol')
        model.load_state_dict(old['model'], strict=True)
        optimizer.load_state_dict(old['optimizer'])
        start_step, records, seen = old['steps'], old['records'], set(old['seen_sample_ids'])
    if steps <= start_step:
        raise ValueError('Total requested steps must exceed the checkpoint step')
    val_text, val_video = encode_batch(provider, val_rows, device)
    before = old['before'] if old else diagnostics(model.eval(), val_text, val_video)
    model.train()
    batch = min(cfg['ground_batch'], len(rows))
    started = time.monotonic()

    def checkpoint(step):
        model.eval()
        after = diagnostics(model, val_text, val_video)
        chance = 1/len(val_rows)
        metrics = after[mode]
        validated = (not after['degenerate'] and min(metrics['v2t_r1'], metrics['t2v_r1']) > chance)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), mode=mode,
                       steps=step, config=cfg, contract=contract, records=records,
                       seen_sample_ids=sorted(seen), before=before, validation=after,
                       grounding_validated=validated, validation_chance_r1=chance,
                       semantic_selectivity_proven=False)
        save_torch(output/'grounding.pt', payload)
        write_jsonl(output/'steps.jsonl', records)
        write_json(output/'report.json', dict(mode=mode, steps=step, new_steps=step-start_step,
                   train_manifest_count=len(load_split(cfg['manifests'], 'train')),
                   effective_training_pool=len(rows), unique_training_samples_seen=len(seen),
                   validation_count=len(val_rows), validation_chance_r1=chance,
                   before=before, after=after, grounding_validated=validated,
                   feature_cache_bytes=0, raw_teacher_tokens_supported=True,
                   seconds=time.monotonic()-started, test_used=False,
                   semantic_selectivity_proven=False, formal_efficacy_evidence=False))
        model.train()
        return payload

    for step in range(start_step, steps):
        selection = random.Random(cfg['seed']+step).sample(rows, batch)
        text, video = encode_batch(provider, selection, device)
        seed_everything(cfg['seed']+20000+step)
        optimizer.zero_grad(set_to_none=True)
        loss, logs = model.loss(text, video, [r['caption_group'] for r in selection],
                                [r['group_id'] for r in selection], mode, cfg['sparse_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite Grounding loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite Grounding gradients')
        optimizer.step()
        seen.update(r['sample_id'] for r in selection)
        record = dict(step=step+1, loss=loss.item(), gradient_norm=norm.item(),
                      sample_ids=[r['sample_id'] for r in selection],
                      **{k: v.item() for k, v in logs.items()})
        records.append(record)
        print(mode, record, flush=True)
        if (step+1) % cfg['checkpoint_every'] == 0:
            checkpoint(step+1)
    result = checkpoint(steps)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--mode', choices=['full', 'said'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--val-limit', type=int, default=16)
    parser.add_argument('--resume')
    args = parser.parse_args()
    train(load_config(args.config), args.mode, args.output, args.steps,
          args.train_limit, args.val_limit, args.resume)
