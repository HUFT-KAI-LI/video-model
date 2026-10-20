import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .alignment import Grounding, HiddenCapture, Projector, distillation_loss
from .cache_features import load_features
from .data import load_split
from .io_utils import sha256, write_json, write_jsonl
from .runtime import (config, forward_fn, load_dit, load_trainable, save_torch, seed_everything,
                      trainable_state, upstream)


def gradient_report(model):
    block_norms = [0.] * 30
    other = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not param.requires_grad or 'lora_' not in name:
            raise AssertionError(f'Unexpected backbone gradient: {name}')
        if name.startswith('blocks.'):
            block_norms[int(name.split('.')[1])] += param.grad.float().square().sum().item()
        else:
            other.append(name)
    return dict(block_norms=[n ** .5 for n in block_norms],
                nonzero_blocks=[i + 1 for i, n in enumerate(block_norms) if n > 0], other=other)


def train(cfg, group, output, steps, device, grounding_path=None, gradient_audit=False):
    upstream()
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing run')
    output.mkdir(parents=True)
    write_json(output / 'config.json', dict(config=cfg, group=group, steps=steps, device=device))
    seed_everything(cfg['seed'])
    rows = load_split(cfg['manifests'], 'train', training=True)
    model = load_dit(cfg, device).train()
    projector = Projector(1536, cfg['dim']).to(device)
    ground = None
    if group != 'A':
        if not grounding_path:
            raise ValueError('B and C require independently pretrained Grounding checkpoints')
        checkpoint = torch.load(grounding_path, map_location='cpu', weights_only=True)
        if checkpoint['mode'] != ('full' if group == 'B' else 'said'):
            raise ValueError('Wrong Grounding control arm')
        if checkpoint['split_lock_sha256'] != sha256(Path(cfg['manifests']) / 'split_lock.json'):
            raise ValueError('Grounding checkpoint belongs to a different split')
        ground = Grounding(video_dim=cfg['teacher_dim'], dim=cfg['dim']).to(device)
        ground.load_state_dict(checkpoint['model'], strict=True)
        ground.eval().requires_grad_(False)
    else:
        projector.requires_grad_(False)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = [p for p in projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params + head_params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = FlowMatchScheduler('Wan')
    scheduler.set_timesteps(1000, training=True)
    native_forward = forward_fn()
    latest = {}

    def fm_forward(dit, latents, timestep, **inputs):
        latest['timestep'] = float(timestep.item())
        nearest = (scheduler.timesteps - timestep.cpu()).abs().argmin()
        latest['sigma'] = float(scheduler.sigmas[nearest])
        return native_forward(dit=dit, latents=latents, timestep=timestep, **inputs)

    pipe = SimpleNamespace(scheduler=scheduler, torch_dtype=torch.bfloat16, device=device,
                           dit=model, in_iteration_models=('dit',), model_fn=fm_forward)
    order_rng = random.Random(cfg['seed'])
    order = []
    while len(order) < steps:
        epoch = list(range(len(rows)))
        order_rng.shuffle(epoch)
        order.extend(epoch)
    initial = trainable_state(model)
    initial_hash = hashlib.sha256(b''.join(p.numpy().tobytes() for p in initial.values())).hexdigest()
    del initial
    records, audits = [], {}
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    for step, index in enumerate(order[:steps]):
        step_start = time.monotonic()
        row = rows[index]
        kinds = ('text', 'vae') if group == 'A' else ('text', 'vae', 'teacher')
        features = load_features(cfg, row, kinds)
        latent = features['vae']['latents'].to(device)
        context = features['text']['context'].to(device)
        seed_everything(cfg['seed'] + 10000 + step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if group == 'A':
                fm = FlowMatchSFTLoss(pipe, input_latents=latent, context=context,
                                      use_gradient_checkpointing=cfg['gradient_checkpointing'])
                distill = fm.new_zeros(())
            else:
                with HiddenCapture(model.blocks[14]) as capture:
                    fm = FlowMatchSFTLoss(pipe, input_latents=latent, context=context,
                                          use_gradient_checkpointing=cfg['gradient_checkpointing'])
                capture.hidden.retain_grad()
                target = ground.target(features['text']['pooled'].to(device),
                                       features['teacher']['pooled'].to(device),
                                       mode='full' if group == 'B' else 'said')
                distill = distillation_loss(projector(capture.hidden), target)
            active_distill = group != 'A' and latest['sigma'] <= cfg['distill_max_sigma']
            total = fm + (cfg['distill_weight'] * distill if active_distill else 0)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite training loss')
        if step == 0 and gradient_audit:
            fm.backward(retain_graph=True)
            audits['fm'] = gradient_report(model)
            audits['fm_head_has_grad'] = any(p.grad is not None for p in head_params)
            assert audits['fm']['nonzero_blocks'] == list(range(1, 31))
            assert not audits['fm_head_has_grad']
            optimizer.zero_grad(set_to_none=True)
            if group != 'A':
                capture.hidden.grad = None
                distill.backward(retain_graph=True)
                audits['distill'] = gradient_report(model)
                audits['distill_h15_grad_norm'] = capture.hidden.grad.norm().item()
                audits['distill_head_grad_norm'] = sum(p.grad.float().square().sum().item()
                                                      for p in head_params if p.grad is not None) ** .5
                assert audits['distill']['nonzero_blocks'] == list(range(1, 16))
                assert audits['distill_h15_grad_norm'] > 0
                assert audits['distill_head_grad_norm'] > 0
                assert all(p.grad is None for p in ground.parameters())
                optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(lora_params + head_params, cfg['gradient_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradients')
        lora_norm = sum(p.grad.float().square().sum().item() for p in lora_params if p.grad is not None) ** .5
        optimizer.step()
        torch.cuda.synchronize(device)
        bucket = 'low' if latest['sigma'] < 1/3 else 'mid' if latest['sigma'] < 2/3 else 'high'
        records.append(dict(step=step + 1, sample_id=row['sample_id'], fm=fm.item(),
                            distill=distill.item(), loss=total.item(), lora_grad_norm=lora_norm,
                            preclip_grad_norm=norm.item(), sigma=latest['sigma'], timestep=latest['timestep'],
                            noise_bucket=bucket, distill_active=active_distill,
                            step_seconds=time.monotonic() - step_start))
        write_jsonl(output / 'steps.jsonl', records)
        print(group, records[-1], flush=True)
    state = trainable_state(model)
    checkpoint = dict(lora=state, projector=projector.state_dict(), optimizer=optimizer.state_dict(),
                      group=group, steps=steps, config=cfg, initial_lora_sha256=initial_hash,
                      split_lock_sha256=sha256(Path(cfg['manifests']) / 'split_lock.json'),
                      grounding_sha256=sha256(grounding_path) if grounding_path else None,
                      cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device))
    save_torch(output / 'checkpoint.pt', checkpoint)
    # Verify adapter serialization and a native prediction roundtrip after the last update.
    restored = torch.load(output / 'checkpoint.pt', map_location='cpu', weights_only=True)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        timestep = torch.tensor([500.], device=device, dtype=torch.bfloat16)
        prediction = native_forward(dit=model, latents=latent, timestep=timestep, context=context)
        with torch.no_grad():
            for p in lora_params:
                p.zero_()
        load_trainable(model, restored['lora'])
        prediction_restored = native_forward(dit=model, latents=latent, timestep=timestep, context=context)
        maximum_error = (prediction - prediction_restored).abs().max().item()
    assert maximum_error == 0, maximum_error
    write_json(output / 'gradient_audit.json', audits)
    write_json(output / 'report.json', dict(group=group, steps=steps, seconds=time.monotonic() - start,
               initial_lora_sha256=initial_hash, device=device,
               peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
               peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
               serialization_prediction_max_error=maximum_error, lora_parameters=sum(p.numel() for p in lora_params),
               student_projector_parameters=sum(p.numel() for p in head_params),
               grounding_frozen=True, effective_loss='FM' if group == 'A' else 'FM + 0.05 * Distill',
               scheme='Staged frozen Grounding; no Grounding loss in DiT optimizer',
               noise_buckets={b: [r['distill'] for r in records if r['noise_bucket'] == b]
                              for b in ('low', 'mid', 'high')},
               oom=False, nan=False, formal_efficacy_evidence=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--grounding')
    parser.add_argument('--device')
    parser.add_argument('--gradient-audit', action='store_true')
    args = parser.parse_args()
    cfg = config(args.config)
    train(cfg, args.group, args.output, args.steps, args.device or cfg['devices'][args.group],
          args.grounding, args.gradient_audit)
