import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from ..alignment import Grounding, HiddenCapture, Projector, distillation_loss
from ..data import load_split
from ..io_utils import sha256, write_json, write_jsonl
from ..runtime import (forward_fn, load_dit, load_trainable, save_torch, seed_everything,
                       trainable_state, upstream)
from ..train import gradient_report
from .config import load_config
from .execution import deterministic_training
from .online import OnlineFeatures, experiment_identity


def sample_index(step, count, seed):
    epoch, position = divmod(step, count)
    order = list(range(count))
    random.Random(seed+epoch).shuffle(order)
    return order[position]


def state_hash(state):
    return hashlib.sha256(b''.join(value.detach().cpu().contiguous().numpy().tobytes()
                                  for value in state.values())).hexdigest()


def require_review(path, identity, allowed_ids):
    if path is None:
        raise ValueError('Non-smoke training requires an explicit human caption/crop/event review approval')
    review = json.loads(Path(path).read_text())
    if (not review.get('approved') or not review.get('reviewer') or not review.get('reviewed_sample_ids') or
            review.get('preprocessing_hash') != identity['preprocessing_hash'] or
            review.get('split_lock_sha256') != identity['split_lock_sha256'] or
            not set(review['reviewed_sample_ids']).issubset(allowed_ids) or
            review.get('rejected_sample_ids')):
        raise ValueError('Human review is absent or belongs to another dataset/preprocessing protocol')


def review_policy(smoke, exploratory, review, identity, allowed_ids):
    if smoke and exploratory:
        raise ValueError('Exploratory full-pilot training is distinct from an engineering smoke run')
    if not smoke and (not exploratory or review is not None):
        require_review(review, identity, allowed_ids)
    return dict(human_review_approved=review is not None and not smoke,
                human_review_pending=bool(exploratory and review is None),
                publication_eligible=not smoke and not exploratory)


def require_grounding(state, group, identity, smoke):
    if state['mode'] != ('full' if group == 'B' else 'said') or state['contract']['identity'] != identity:
        raise ValueError('Rejecting wrong arm, stale ViT-Small, teacher/preprocessing/split mismatch')
    if not smoke and not state['grounding_validated']:
        raise ValueError('Grounding is unvalidated/near random; only an explicitly marked smoke run is allowed')


@deterministic_training
def train(cfg, group, output, steps, grounding_path=None, resume=None, smoke=False,
          review=None, gradient_audit=False, provider=None, exploratory=False):
    if group not in ('A', 'B', 'C') or steps < 1:
        raise ValueError('Invalid arm or optimizer step count')
    upstream()
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    from diffsynth.diffusion.loss import FlowMatchSFTLoss
    identity = experiment_identity(cfg)
    allowed_ids = {r['sample_id'] for split in ('train', 'val')
                   for r in load_split(cfg['manifests'], split)}
    review_status = review_policy(smoke, exploratory, review, identity, allowed_ids)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError('Use a fresh output or an explicit resume checkpoint')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg['seed'])
    device = cfg['devices']['student']
    rows = load_split(cfg['manifests'], 'train', training=True)
    provider = provider or OnlineFeatures(cfg)
    model = load_dit(cfg, device).train()
    projector = Projector(1536, cfg['dim']).to(device)
    ground = None
    if group == 'A':
        projector.requires_grad_(False)
    else:
        if grounding_path is None:
            raise ValueError('B/C require a freshly trained ViT-Base Grounding checkpoint')
        ground_state = torch.load(grounding_path, map_location='cpu', weights_only=True)
        require_grounding(ground_state, group, identity, smoke)
        ground = Grounding(video_dim=768, dim=cfg['dim']).to(device)
        ground.load_state_dict(ground_state['model'], strict=True)
        ground.eval().requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    head_parameters = [p for p in projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters+head_parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    initial_hash = state_hash(trainable_state(model))
    contract = dict(identity=identity, group=group, smoke=smoke,
                    grounding_sha256=sha256(grounding_path) if grounding_path else None)
    if exploratory:
        contract.update(exploratory=True, review_status=review_status)
    start_step, records, audits = 0, [], {}
    if resume:
        previous = torch.load(resume, map_location='cpu', weights_only=True)
        if previous['contract'] != contract:
            raise ValueError('Resume changed arm, preprocessing, split, Grounding, smoke status, or optimizer protocol')
        load_trainable(model, previous['lora'])
        projector.load_state_dict(previous['projector'], strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        initial_hash, start_step = previous['initial_lora_sha256'], previous['steps']
        records, audits = previous['records'], previous['gradient_audit']
    if steps <= start_step:
        raise ValueError('Requested total steps must exceed the checkpoint step')
    scheduler = FlowMatchScheduler('Wan')
    scheduler.set_timesteps(1000, training=True)
    native_forward = forward_fn()
    latest = {}

    def fm_forward(dit, latents, timestep, **kwargs):
        index = (scheduler.timesteps-timestep.cpu()).abs().argmin()
        latest.update(timestep=float(timestep.item()), sigma=float(scheduler.sigmas[index]))
        return native_forward(dit=dit, latents=latents, timestep=timestep, **kwargs)

    pipe = SimpleNamespace(scheduler=scheduler, torch_dtype=torch.bfloat16, device=device,
                           dit=model, in_iteration_models=('dit',), model_fn=fm_forward)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(step):
        state = trainable_state(model)
        payload = dict(lora=state, projector=projector.state_dict(), optimizer=optimizer.state_dict(),
                       initial_lora_sha256=initial_hash, final_lora_sha256=state_hash(state),
                       group=group, steps=step, config=cfg, contract=contract,
                       records=records, gradient_audit=audits)
        save_torch(output/'checkpoint.pt', payload)
        restored = torch.load(output/'checkpoint.pt', map_location='cpu', weights_only=True)
        if state_hash(restored['lora']) != payload['final_lora_sha256']:
            raise ValueError('Adapter serialization checksum mismatch')
        unique = len({r['sample_id'] for r in records})
        write_jsonl(output/'steps.jsonl', records)
        write_json(output/'gradient_audit.json', audits)
        write_json(output/'report.json', dict(group=group, steps=step, new_steps=step-start_step,
                   train_manifest_count=len(rows), unique_training_samples_seen=unique,
                   sample_presentations=len(records), complete_data_passes=step//len(rows),
                   equivalent_epochs=step/len(rows), initial_lora_sha256=initial_hash,
                   seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                   preprocessing_hash=identity['preprocessing_hash'], feature_cache_bytes=0,
                   smoke=smoke, test_used=False, formal_efficacy_evidence=False,
                   effective_loss='FM' if group == 'A' else f'FM + {cfg["distill_weight"]} * {"Full" if group == "B" else "Said"}',
                   native_fm_function='diffsynth.diffusion.loss.FlowMatchSFTLoss (unchanged)',
                   exploratory=exploratory, review_status=review_status,
                   grounding_frozen=True, teacher_frozen=True, raw_teacher_tokens_supported=True))
        return payload

    for step in range(start_step, steps):
        row = rows[sample_index(step, len(rows), cfg['seed'])]
        begin = time.monotonic()
        kinds = ('text', 'vae') if group == 'A' else ('text', 'vae', 'teacher')
        features = provider(row, kinds)
        latent = features['vae']['latents'].to(device)
        context = features['text']['context'].to(device)
        seed_everything(cfg['seed']+10000+step)
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
                                       'full' if group == 'B' else 'said')
                distill = distillation_loss(projector(capture.hidden), target)
            active = group != 'A' and latest['sigma'] <= cfg['distill_max_sigma']
            total = fm+(cfg['distill_weight']*distill if active else 0)
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite FM/Said loss')
        if step == start_step and gradient_audit:
            fm.backward(retain_graph=True)
            audits['fm'] = gradient_report(model)
            audits['fm_projector_has_grad'] = any(p.grad is not None for p in head_parameters)
            if audits['fm']['nonzero_blocks'] != list(range(1, 31)) or audits['fm_projector_has_grad']:
                raise AssertionError('FM gradient contract changed')
            optimizer.zero_grad(set_to_none=True)
            if group != 'A':
                distill.backward(retain_graph=True)
                audits['distill'] = gradient_report(model)
                audits['h15_gradient_norm'] = capture.hidden.grad.norm().item()
                audits['projector_gradient_norm'] = sum(p.grad.float().square().sum().item()
                                                        for p in head_parameters if p.grad is not None)**.5
                if (audits['distill']['nonzero_blocks'] != list(range(1, 16)) or
                        audits['h15_gradient_norm'] <= 0 or audits['projector_gradient_norm'] <= 0 or
                        any(p.grad is not None for p in ground.parameters())):
                    raise AssertionError('H15/frozen Grounding gradient contract changed')
                optimizer.zero_grad(set_to_none=True)
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters+head_parameters, cfg['gradient_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite trainable gradient')
        optimizer.step()
        torch.cuda.synchronize(device)
        record = dict(step=step+1, sample_id=row['sample_id'], epoch=step//len(rows),
                      noise_seed=cfg['seed']+10000+step, timestep=latest['timestep'], sigma=latest['sigma'],
                      fm=fm.item(), distill=distill.item(), distill_active=active,
                      loss=total.item(), gradient_norm=norm.item(), seconds=time.monotonic()-begin,
                      latent_sha256=state_hash({'latent': latent.float()}),
                      sampling=features['sampling'])
        records.append(record)
        print(group, {k: v for k, v in record.items() if k != 'sampling'}, flush=True)
        if (step+1) % cfg['checkpoint_every'] == 0:
            checkpoint(step+1)
    return checkpoint(steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--group', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, required=True, help='Total optimizer steps, including any resumed steps')
    parser.add_argument('--grounding')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--approved-review')
    parser.add_argument('--gradient-audit', action='store_true')
    parser.add_argument('--exploratory', action='store_true', help='Explicit non-publication pilot; human review may be pending, Grounding validation still required')
    args = parser.parse_args()
    train(load_config(args.config), args.group, args.output, args.steps, args.grounding,
          args.resume, args.smoke, args.approved_review, args.gradient_audit,
          exploratory=args.exploratory)
