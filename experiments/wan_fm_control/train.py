"""Four-rank FM-only LoRA control with the same sample/noise schedule as C."""
import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..wan_said.data import load_split
from ..wan_said.io_utils import sha256, write_json, write_jsonl
from ..wan_said.runtime import (forward_fn, load_dit, load_trainable, save_torch,
                               seed_everything, trainable_state, upstream)
from ..wan_said.train import gradient_report
from ..wan_said.pilot_v2.config import load_config
from ..wan_said.pilot_v2.execution import deterministic_attention
from ..wan_said.pilot_v2.online import experiment_identity
from ..wan_said.pilot_v2.parallel.train import LocalFeatures, assignment
from ..wan_said.pilot_v2.train import review_policy, sample_index, state_hash


def check_reference(saved, cfg, identity, count):
    if (saved['group'] != 'C' or saved['contract']['smoke'] or
            saved['contract']['identity'] != identity or saved['config'] != cfg or
            saved['steps'] != math.ceil(count/4) or
            saved['contract']['parallel'] != dict(backend='torch_ddp', world_size=4,
                local_batch=1, global_batch=4, tail='zero_weight_padding', epochs=1)):
        raise ValueError('FM control requires the completed matching four-GPU C run')
    rows = load_split(cfg['manifests'], 'train', training=True)
    expected = [dict(sample_id=rows[sample_index(i, count, cfg['seed'])]['sample_id'],
                     noise_seed=cfg['seed']+10000+i, rank=i % 4, step=i//4+1)
                for i in range(count)]
    actual = [{key: r[key] for key in expected[0]} for r in saved['records']]
    if actual != expected:
        raise ValueError('C training order/noise seeds do not match the control protocol')


class Objective(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from diffsynth.diffusion.flow_match import FlowMatchScheduler
        self.cfg, self.device = cfg, cfg['devices']['student']
        self.dit = load_dit(cfg, self.device).train()
        self.scheduler = FlowMatchScheduler('Wan')
        self.scheduler.set_timesteps(1000, training=True)
        self.native, self.latest = forward_fn(), {}

    def forward(self, features):
        from diffsynth.diffusion.loss import FlowMatchSFTLoss

        def forward(dit, latents, timestep, **kwargs):
            index = (self.scheduler.timesteps-timestep.cpu()).abs().argmin()
            self.latest.update(timestep=float(timestep.item()), sigma=float(self.scheduler.sigmas[index]))
            return self.native(dit=dit, latents=latents, timestep=timestep, **kwargs)

        pipe = SimpleNamespace(scheduler=self.scheduler, torch_dtype=torch.bfloat16,
            device=self.device, dit=self.dit, in_iteration_models=('dit',), model_fn=forward)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return FlowMatchSFTLoss(pipe, input_latents=features['vae']['latents'].to(self.device),
                context=features['text']['context'].to(self.device),
                use_gradient_checkpointing=self.cfg['gradient_checkpointing'])


def train(cfg, output, reference, resume=None, smoke_steps=None):
    upstream()
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('The matched FM protocol requires exactly four ranks')
    torch.cuda.set_device(rank)
    free, capacity = torch.cuda.mem_get_info(rank)
    budget = min(20*2**30, free-2*2**30)
    if budget < 10*2**30:
        raise RuntimeError(f'GPU {rank} lacks free VRAM; unrelated processes will not be stopped')
    torch.cuda.set_per_process_memory_fraction(budget/capacity, rank)
    local = copy.deepcopy(cfg)
    local['devices'] = {key: f'cuda:{rank}' for key in cfg['devices']}
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    output = Path(output)
    try:
        with deterministic_attention(cfg['deterministic_attention']):
            identity = experiment_identity(cfg)
            rows = load_split(cfg['manifests'], 'train', training=True)
            reference_state = torch.load(reference, map_location='cpu', weights_only=True)
            check_reference(reference_state, cfg, identity, len(rows))
            reference_initial = reference_state['initial_lora_sha256']
            del reference_state
            smoke = smoke_steps is not None
            if smoke and not 1 <= smoke_steps <= math.ceil(len(rows)/world):
                raise ValueError('Invalid smoke step count')
            allowed = {r['sample_id'] for s in ('train', 'val') for r in load_split(cfg['manifests'], s)}
            policy = review_policy(smoke, not smoke, None, identity, allowed)
            if output.exists() and not resume:
                raise FileExistsError('Use a fresh output or explicit resume checkpoint')
            dist.barrier()
            if rank == 0:
                output.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            contract = dict(identity=identity, group='A', smoke=smoke, grounding_sha256=None,
                reference_C_sha256=sha256(reference), effective_loss='FM',
                parallel=dict(backend='torch_ddp', world_size=4, local_batch=1,
                              global_batch=4, tail='zero_weight_padding', epochs=1))
            if not smoke:
                contract.update(exploratory=True, review_status=policy)
            seed_everything(cfg['seed'])
            objective = Objective(local)
            parameters = [p for p in objective.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(parameters, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            initial_hash = state_hash(trainable_state(objective.dit))
            if initial_hash != reference_initial:
                raise ValueError('Initial FM-only LoRA differs from C; refusing an unmatched control')
            records, audits, start = [], {}, 0
            if resume:
                saved = torch.load(resume, map_location='cpu', weights_only=True)
                if saved['contract'] != contract:
                    raise ValueError('FM resume contract changed')
                load_trainable(objective.dit, saved['lora'])
                optimizer.load_state_dict(saved['optimizer'])
                records, audits, start = saved['records'], saved['gradient_audit'], saved['steps']
                del saved
            steps = smoke_steps if smoke else math.ceil(len(rows)/world)
            if start >= steps:
                raise ValueError('No remaining optimizer updates')
            provider = LocalFeatures(local, offload_text=budget < 19*2**30)
            wrapped = None
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(rank)

            def checkpoint(step):
                state = trainable_state(objective.dit)
                hashes = [None]*world
                dist.all_gather_object(hashes, state_hash(state))
                if len(set(hashes)) != 1:
                    raise AssertionError('FM-only LoRA diverged between ranks')
                if rank == 0:
                    save_torch(output/'checkpoint.pt', dict(lora=state, optimizer=optimizer.state_dict(),
                        initial_lora_sha256=initial_hash, final_lora_sha256=hashes[0], group='A',
                        steps=step, config=cfg, contract=contract, records=records, gradient_audit=audits))
                    write_jsonl(output/'steps.jsonl', records)
                    write_json(output/'gradient_audit.json', audits)
                    write_json(output/'report.json', dict(group='A', steps=step, complete=step == steps,
                        full_epoch_complete=not smoke and step == steps, smoke=smoke,
                        global_batch=4, world_size=4, train_manifest_count=len(rows),
                        unique_training_samples_seen=len({r['sample_id'] for r in records}),
                        sample_presentations=len(records), equivalent_epochs=len(records)/len(rows),
                        effective_loss='FM', teacher_used=False, grounding_used=False, projector_used=False,
                        feature_cache_bytes=0, test_used=False, exploratory=not smoke,
                        publication_eligible=False, native_fm_function='diffsynth.diffusion.loss.FlowMatchSFTLoss (unchanged)',
                        initial_lora_sha256=initial_hash, reference_C_sha256=contract['reference_C_sha256'],
                        seconds=time.monotonic()-started, new_steps=step-start, last_rank_metrics=rank_metrics,
                        synchronized_parameter_sha256=hashes))
                dist.barrier()

            print(json.dumps(dict(rank=rank, phase='loading', budget_gib=budget/2**30,
                                  text_cpu_offload=provider.offload_text, steps=steps, group='A')), flush=True)
            for step in range(start, steps):
                begin = time.monotonic()
                position, scale = assignment(step, len(rows), world, rank)
                row = rows[sample_index(position, len(rows), cfg['seed'])]
                features = provider(row, ('text', 'vae'))
                noise_seed = cfg['seed']+10000+position
                if wrapped is None:
                    seed_everything(noise_seed)
                    loss = objective(features)
                    loss.backward()
                    audit = gradient_report(objective.dit)
                    if audit['nonzero_blocks'] != list(range(1, 31)):
                        raise AssertionError('Native FM must reach all 30 DiT blocks')
                    gathered = [None]*world
                    dist.all_gather_object(gathered, audit)
                    audits = {str(i): a for i, a in enumerate(gathered)}
                    objective.zero_grad(set_to_none=True)
                    del loss
                    wrapped = DDP(objective, device_ids=[rank], broadcast_buffers=False,
                                  find_unused_parameters=False, gradient_as_bucket_view=True)
                seed_everything(noise_seed)
                optimizer.zero_grad(set_to_none=True)
                loss = wrapped(features)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite FM loss')
                (loss*scale).backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['gradient_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite FM gradient')
                optimizer.step()
                torch.cuda.synchronize(rank)
                record = dict(step=step+1, sample_id=row['sample_id'], epoch=0, rank=rank,
                    noise_seed=noise_seed, **objective.latest, fm=loss.item(), loss=loss.item(),
                    distill=0., gradient_norm=norm.item(), supervised=scale > 0,
                    seconds=time.monotonic()-begin, sampling=features['sampling'],
                    peak_allocated_gib=torch.cuda.max_memory_allocated(rank)/2**30,
                    budget_gib=budget/2**30, text_cpu_offload=provider.offload_text)
                rank_metrics = [None]*world
                dist.all_gather_object(rank_metrics, record)
                if rank == 0:
                    records.extend(r for r in rank_metrics if r['supervised'])
                    print('DDP_A', json.dumps(dict(step=step+1, samples=len(records),
                        fm=sum(r['fm'] for r in rank_metrics if r['supervised'])/sum(r['supervised'] for r in rank_metrics),
                        seconds=max(r['seconds'] for r in rank_metrics))), flush=True)
                del features, loss
                if step == start or (step+1) % cfg['checkpoint_every'] == 0:
                    checkpoint(step+1)
            checkpoint(steps)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--smoke-steps', type=int)
    args = parser.parse_args()
    train(load_config(args.config), args.output, args.reference, args.resume, args.smoke_steps)
