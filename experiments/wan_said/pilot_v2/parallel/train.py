"""Four-rank DDP C pilot, using the unchanged native FM and shared SAID losses."""
import argparse
import copy
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ...alignment import Grounding, HiddenCapture, Projector, distillation_loss
from ...data import load_split
from ...io_utils import sha256, write_json, write_jsonl
from ...runtime import (forward_fn, load_dit, load_trainable, save_torch,
                        seed_everything, trainable_state, upstream)
from ...train import gradient_report
from ..config import load_config
from ..execution import deterministic_attention
from ..online import OnlineFeatures, experiment_identity
from ..train import require_grounding, review_policy, sample_index, state_hash


def assignment(step, count, workers, rank):
    """Last incomplete batch has zero-weight padding, never duplicate supervision."""
    if not 0 <= rank < workers or not 0 <= step < math.ceil(count/workers):
        raise ValueError('Invalid epoch position or rank')
    position = step*workers+rank
    active = position < count
    actual = min(workers, count-step*workers)
    return min(position, count-1), workers/actual if active else 0.0


class LocalFeatures(OnlineFeatures):
    def __init__(self, cfg, offload_text=False):
        super().__init__(cfg)
        self.offload_text = offload_text

    @torch.no_grad()
    def __call__(self, row, kinds=('text', 'teacher', 'vae')):
        if not self.offload_text:
            result = super().__call__(row, kinds)
        else:
            if self.text is not None:
                self.text.to(self.cfg['devices']['text'])
            result = super().__call__(row, ('text',))
            self.text.to('cpu')
            torch.cuda.empty_cache()
            result.update(super().__call__(row, tuple(k for k in kinds if k != 'text')))
        # Retain only this batch's tensors, not a process-lifetime per-call history.
        self.calls.clear()
        return result


class Objective(nn.Module):
    def __init__(self, cfg, identity, grounding, smoke):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.cfg, self.device = cfg, cfg['devices']['student']
        self.dit = load_dit(cfg, self.device).train()
        self.projector = Projector(1536, cfg['dim']).to(self.device)
        saved = torch.load(grounding, map_location='cpu', weights_only=True)
        require_grounding(saved, 'C', identity, smoke)
        self.ground = Grounding(video_dim=768, dim=cfg['dim']).to(self.device)
        self.ground.load_state_dict(saved['model'], strict=True)
        self.ground.eval().requires_grad_(False)
        self.scheduler = FlowMatchScheduler('Wan')
        self.scheduler.set_timesteps(1000, training=True)
        self.native = forward_fn()
        self.latest = {}

    def forward(self, features):
        from diffsynth.diffusion.loss import FlowMatchSFTLoss

        def forward(dit, latents, timestep, **kwargs):
            index = (self.scheduler.timesteps-timestep.cpu()).abs().argmin()
            self.latest.update(timestep=float(timestep.item()), sigma=float(self.scheduler.sigmas[index]))
            return self.native(dit=dit, latents=latents, timestep=timestep, **kwargs)

        pipe = SimpleNamespace(scheduler=self.scheduler, torch_dtype=torch.bfloat16, device=self.device,
                               dit=self.dit, in_iteration_models=('dit',), model_fn=forward)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with HiddenCapture(self.dit.blocks[14]) as capture:
                fm = FlowMatchSFTLoss(pipe, input_latents=features['vae']['latents'].to(self.device),
                                      context=features['text']['context'].to(self.device),
                                      use_gradient_checkpointing=self.cfg['gradient_checkpointing'])
            target = self.ground.target(features['text']['pooled'].to(self.device),
                                       features['teacher']['pooled'].to(self.device), 'said')
            said = distillation_loss(self.projector(capture.hidden), target)
            active = self.latest['sigma'] <= self.cfg['distill_max_sigma']
            loss = fm+(self.cfg['distill_weight']*said if active else 0)
        return loss, fm, said


def audit(objective, features, seed):
    seed_everything(seed)
    total, fm, said = objective(features)
    fm.backward(retain_graph=True)
    result = dict(fm=gradient_report(objective.dit),
                  fm_projector_has_grad=any(p.grad is not None for p in objective.projector.parameters()))
    objective.zero_grad(set_to_none=True)
    said.backward()
    result['distill'] = gradient_report(objective.dit)
    result['projector_gradient_norm'] = sum(p.grad.float().square().sum().item()
                                           for p in objective.projector.parameters() if p.grad is not None)**.5
    if (result['fm']['nonzero_blocks'] != list(range(1, 31)) or result['fm_projector_has_grad'] or
            result['distill']['nonzero_blocks'] != list(range(1, 16)) or
            result['projector_gradient_norm'] <= 0 or any(p.grad is not None for p in objective.ground.parameters())):
        raise AssertionError('Native FM / H15 SAID / frozen Grounding gradient contract changed')
    objective.zero_grad(set_to_none=True)
    del total, fm, said
    return result


def train(cfg, output, grounding, resume=None, smoke=False, exploratory=False, review=None,
          max_steps=None, gradient_audit=False):
    upstream()
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('This protocol requires exactly four ranks, one per visible GPU')
    torch.cuda.set_device(rank)
    device = f'cuda:{rank}'
    free, capacity = torch.cuda.mem_get_info(rank)
    budget = min(20*2**30, free-2*2**30)
    if budget < 10*2**30:
        raise RuntimeError(f'GPU {rank} has insufficient free VRAM; other processes will not be stopped')
    torch.cuda.set_per_process_memory_fraction(budget/capacity, rank)
    offload = budget < 19*2**30
    local = copy.deepcopy(cfg)
    local['devices'] = {key: device for key in cfg['devices']}
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    output = Path(output)
    try:
        with deterministic_attention(cfg['deterministic_attention']):
            identity = experiment_identity(cfg)
            rows = load_split(cfg['manifests'], 'train', training=True)
            allowed = {r['sample_id'] for s in ('train', 'val') for r in load_split(cfg['manifests'], s)}
            policy = review_policy(smoke, exploratory, review, identity, allowed)
            if output.exists() and not resume:
                raise FileExistsError('Use a fresh DDP output or an explicit checkpoint')
            # Only rank zero may create the output before the shared existence check.
            dist.barrier()
            if rank == 0:
                output.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            contract = dict(identity=identity, group='C', smoke=smoke, grounding_sha256=sha256(grounding),
                            parallel=dict(backend='torch_ddp', world_size=world, local_batch=1,
                                          global_batch=world, tail='zero_weight_padding', epochs=1))
            if exploratory:
                contract.update(exploratory=True, review_status=policy)
            seed_everything(cfg['seed'])
            objective = Objective(local, identity, grounding, smoke)
            parameters = [p for p in objective.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            initial_hash = state_hash(trainable_state(objective.dit))
            records, audits, start = [], {}, 0
            if resume:
                saved = torch.load(resume, map_location='cpu', weights_only=True)
                if saved['contract'] != contract:
                    raise ValueError('DDP resume changed split, global batch, world size, loss, or Grounding')
                load_trainable(objective.dit, saved['lora'])
                objective.projector.load_state_dict(saved['projector'], strict=True)
                optimizer.load_state_dict(saved['optimizer'])
                records, audits, start = saved['records'], saved['gradient_audit'], saved['steps']
                initial_hash = saved['initial_lora_sha256']
                del saved
            steps = math.ceil(len(rows)/world)
            if max_steps is not None:
                if not smoke:
                    raise ValueError('Partial steps require --smoke; real training always covers one full epoch')
                steps = min(steps, max_steps)
            if start >= steps:
                raise ValueError('Requested steps must exceed the resumed checkpoint')
            provider = LocalFeatures(local, offload)
            print(json.dumps(dict(rank=rank, phase='loading', budget_gib=budget/2**30,
                                  text_cpu_offload=offload, global_batch=world, steps=steps)), flush=True)
            wrapped = None
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(rank)

            def checkpoint(step):
                state = trainable_state(objective.dit)
                local_hash = dict(lora=state_hash(state), projector=state_hash(objective.projector.state_dict()))
                hashes = [None]*world
                dist.all_gather_object(hashes, local_hash)
                if any(value != hashes[0] for value in hashes):
                    raise AssertionError('DDP ranks diverged in LoRA or projector weights')
                if rank == 0:
                    save_torch(output/'checkpoint.pt', dict(lora=state, projector=objective.projector.state_dict(),
                        optimizer=optimizer.state_dict(), initial_lora_sha256=initial_hash,
                        final_lora_sha256=state_hash(state), group='C', steps=step, config=cfg,
                        contract=contract, records=records, gradient_audit=audits))
                    unique = len({r['sample_id'] for r in records})
                    write_jsonl(output/'steps.jsonl', records)
                    write_json(output/'gradient_audit.json', audits)
                    write_json(output/'report.json', dict(group='C', steps=step, new_steps=step-start,
                        optimizer_steps_per_epoch=math.ceil(len(rows)/world), global_batch=world, local_batch=1,
                        world_size=world, train_manifest_count=len(rows), unique_training_samples_seen=unique,
                        sample_presentations=len(records), complete_data_passes=len(records)//len(rows),
                        equivalent_epochs=len(records)/len(rows), seconds=time.monotonic()-started,
                        initial_lora_sha256=initial_hash, preprocessing_hash=identity['preprocessing_hash'],
                        feature_cache_bytes=0, test_used=False, smoke=smoke, exploratory=exploratory,
                        review_status=policy, grounding_frozen=True, teacher_frozen=True,
                        effective_loss=f'FM + {cfg["distill_weight"]} * Said',
                        native_fm_function='diffsynth.diffusion.loss.FlowMatchSFTLoss (unchanged)',
                        formal_efficacy_evidence=False, last_rank_metrics=rank_metrics,
                        synchronized_parameter_sha256=hashes))
                dist.barrier()

            for step in range(start, steps):
                begin = time.monotonic()
                position, scale = assignment(step, len(rows), world, rank)
                row = rows[sample_index(position, len(rows), cfg['seed'])]
                features = provider(row)
                noise_seed = cfg['seed']+10000+position
                if step == start and gradient_audit:
                    # Run the two independent gradient checks before registering DDP's reducer.
                    local_audit = audit(objective, features, noise_seed)
                    all_audits = [None]*world
                    dist.all_gather_object(all_audits, local_audit)
                    audits = {str(i): value for i, value in enumerate(all_audits)}
                if wrapped is None:
                    wrapped = DDP(objective, device_ids=[rank], broadcast_buffers=False,
                                  find_unused_parameters=False, gradient_as_bucket_view=True)
                seed_everything(noise_seed)
                optimizer.zero_grad(set_to_none=True)
                total, fm, said = wrapped(features)
                if not torch.isfinite(total):
                    raise FloatingPointError('Nonfinite distributed FM/SAID loss')
                (total*scale).backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite DDP gradient')
                optimizer.step()
                torch.cuda.synchronize(rank)
                record = dict(step=step+1, sample_id=row['sample_id'], epoch=0, rank=rank,
                              noise_seed=noise_seed, **objective.latest, fm=fm.item(), distill=said.item(),
                              loss=total.item(), gradient_norm=norm.item(), seconds=time.monotonic()-begin,
                              supervised=scale > 0, sampling=features['sampling'],
                              peak_allocated_gib=torch.cuda.max_memory_allocated(rank)/2**30,
                              reserved_gib=torch.cuda.memory_reserved(rank)/2**30,
                              budget_gib=budget/2**30, text_cpu_offload=offload)
                rank_metrics = [None]*world
                dist.all_gather_object(rank_metrics, record)
                if rank == 0:
                    records.extend(r for r in rank_metrics if r['supervised'])
                    print('DDP_C', json.dumps(dict(step=step+1, samples=len(records),
                        loss=sum(r['loss'] for r in rank_metrics if r['supervised'])/sum(r['supervised'] for r in rank_metrics),
                        seconds=max(r['seconds'] for r in rank_metrics),
                        peak_gib=[round(r['peak_allocated_gib'], 2) for r in rank_metrics])), flush=True)
                del features, total, fm, said
                if step == start or (step+1) % cfg['checkpoint_every'] == 0:
                    checkpoint(step+1)
            checkpoint(steps)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--grounding', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--exploratory', action='store_true')
    parser.add_argument('--approved-review')
    parser.add_argument('--gradient-audit', action='store_true')
    args = parser.parse_args()
    train(load_config(args.config), args.output, args.grounding, args.resume, args.smoke,
          args.exploratory, args.approved_review, args.max_steps, args.gradient_audit)
