import argparse
import json
import random
import time
from pathlib import Path

import torch

from .alignment import Grounding
from .cache_features import load_features
from .data import load_split
from .io_utils import sha256, write_json, write_jsonl
from .runtime import ROOT, config, save_torch, seed_everything


def tensors(cfg, split, device):
    rows = load_split(cfg['manifests'], split, training=(split == 'train'))
    features = [load_features(cfg, row, ('text', 'teacher')) for row in rows]
    text = torch.cat([f['text']['pooled'] for f in features]).to(device)
    video = torch.cat([f['teacher']['pooled'] for f in features]).to(device)
    return rows, text, video


@torch.no_grad()
def diagnostics(model, text, video):
    labels = torch.arange(len(text), device=text.device)
    result = {}
    for mode in ('said', 'full', 'random', 'shuffled_text'):
        scores, _ = model.scores(text, video, mode)
        result[mode] = dict(v2t_r1=(scores.argmax(1) == labels).float().mean().item(),
                            t2v_r1=(scores.argmax(0) == labels).float().mean().item(),
                            diagonal_similarity=scores.diag().mean().item() * .07)
    t, v = model.encode(text, video)
    mask, soft = model.gate(t, v)
    changed, _ = model.gate(t.roll(1, 0), v)
    changed_video, _ = model.gate(t, v.roll(1, 0))
    mean = soft.mean().item()
    text_delta = (mask - changed).abs().mean().item()
    result.update(soft_keep=mean, hard_keep=(soft > .5).float().mean().item(),
                  prompt_delta_l1=text_delta, video_delta_l1=(mask - changed_video).abs().mean().item(),
                  channel_std=soft.std(0).mean().item(),
                  degenerate=mean < .1 or mean > .9 or text_delta < 1e-4)
    # This detects a failure mode, not evidence of fine-grained semantic selection.
    result['semantic_selectivity_proven'] = False
    return result


def train(cfg, mode, output, steps, device):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    output.mkdir(parents=True)
    seed_everything(cfg['seed'])
    rows, text, video = tensors(cfg, 'train', device)
    val_rows, val_text, val_video = tensors(cfg, 'val', device)
    model = Grounding(video_dim=cfg['teacher_dim'], dim=cfg['dim']).to(device)
    if mode == 'full':
        model.gate.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg['ground_lr'], weight_decay=cfg['weight_decay'])
    count = min(cfg['ground_batch'], len(rows))
    if count < 2:
        raise ValueError('No valid contrastive negatives')
    rng = random.Random(cfg['seed'])
    records = []
    before = diagnostics(model, val_text, val_video)
    start = time.monotonic()
    for step in range(steps):
        ids = rng.sample(range(len(rows)), count)
        optimizer.zero_grad(set_to_none=True)
        loss, logs = model.loss(text[ids], video[ids], [rows[i]['caption_group'] for i in ids],
                                [rows[i]['group_id'] for i in ids], mode, cfg['sparse_weight'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Grounding loss is nonfinite')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Grounding gradients are nonfinite')
        optimizer.step()
        records.append(dict(step=step + 1, loss=loss.item(), gradient_norm=norm.item(),
                            **{k: v.item() for k, v in logs.items()}))
        if (step + 1) % 50 == 0:
            print(mode, records[-1], flush=True)
    model.eval()
    after = diagnostics(model, val_text, val_video)
    checkpoint = dict(model={n: p.detach().cpu() for n, p in model.state_dict().items()}, mode=mode,
                      config=cfg, steps=steps,
                      split_lock_sha256=sha256(Path(cfg['manifests']) / 'split_lock.json'),
                      before=before, validation=after)
    save_torch(output / 'grounding.pt', checkpoint)
    write_jsonl(output / 'steps.jsonl', records)
    write_json(output / 'report.json', dict(mode=mode, steps=steps, batch=count, train_count=len(rows),
               val_count=len(val_rows), seconds=time.monotonic() - start, before=before, after=after,
               peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
               scheme='Grounding pretraining only; no DiT, no test features, no gradient accumulation negatives',
               cost_note='These preprocessing/optimization costs are additional to DiT training.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['said', 'full'], required=True)
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--device')
    args = parser.parse_args()
    cfg = config(args.config)
    train(cfg, args.mode, args.output, args.steps or cfg['ground_steps'], args.device or cfg['devices']['grounding'])
