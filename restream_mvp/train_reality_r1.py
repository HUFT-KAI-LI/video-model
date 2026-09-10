"""Dedicated bounded R1-A matched training: frozen routing/backbones, fresh adapter.

Resume accepts only this trainer's same-branch R1 checkpoints with identical
input/config provenance. R0 warm starts are never accepted.
"""
import argparse
import json
from pathlib import Path
import torch
from restream.reality_data import write_json
from restream.reality_encoder import RealityEncoder
from restream.reality_paired import load_global_constant
from restream.reality_runtime import read_reality_config, make_cache, make_dataset, make_memory
from restream.reality_r1_training import (TRAIN_BRANCHES, PreparedInputs, evaluate, optimizer_update,
                                         r1_loss, run_signature, schedule_entry, state_digest, validate_training_config)
from restream.runtime import ROOT, load_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reality_memory_r1_matched.yaml')
    parser.add_argument('--branch',choices=TRAIN_BRANCHES,required=True)
    parser.add_argument('--max-updates',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Local checkpoint directory, never committed')
    parser.add_argument('--report',type=Path,required=True,help='Reviewable metrics/provenance directory')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--evaluate-dev',action='store_true')
    args = parser.parse_args()
    if not args.resume and args.max_updates > 2:
        parser.error('First run the 2-update sanity, then resume its R1 checkpoint')
    config = read_reality_config(args.config)
    validate_training_config(config)
    if not 1 <= args.max_updates <= config['train']['max_updates']:
        parser.error('Requested update budget exceeds the fixed configuration')
    if not torch.cuda.is_available():
        raise RuntimeError('Real R1 training requires CUDA')
    device = torch.device('cuda')
    cache = make_cache(config)
    signature = run_signature(config,cache,args.branch)
    smoke_path = ROOT/'validation/reality_memory/r1_matched/zero_update_smoke.json'
    smoke = json.loads(smoke_path.read_text())
    if (smoke.get('status') != 'passed' or smoke['optimizer_steps'] != 0
            or smoke['train_manifest_sha256'] != signature['manifests']['train']
            or smoke['test_eligibility_sha256'] != signature['test_seal']['report_sha256']
            or smoke['adapter_seed'] != config['seed']):
        raise ValueError('A matched zero-update smoke on the sealed inputs must pass first')
    ds = make_dataset(config,'train',cache)
    dev = make_dataset(config,'val',cache)
    saved = None
    if args.resume:
        saved = torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved.get('stage') != 'r1a_matched' or saved.get('signature') != signature:
            raise ValueError('Resume requires identical R1 branch/config/data/provenance; R0 warm start forbidden')
        if saved['optimizer_steps'] >= args.max_updates:
            raise ValueError('Resume must extend effective update budget')
    elif args.output.exists() or args.report.exists():
        raise FileExistsError('Fresh run requires new output and report directories')
    args.output.mkdir(parents=True,exist_ok=True)
    args.report.mkdir(parents=True,exist_ok=True)
    # Publish deterministic design before any optimizer/Dev computation.
    from restream.reality_selection import select_targets
    design = {'signature':signature,'dev_indices':select_targets(dev.rows,config['eval']['cases'],config['eval']['target_seed']),
              'seed':config['seed'],'branches':list(TRAIN_BRANCHES),'scope':'2-step then 10-update Dev mechanism experiment; no Test model access'}
    if (args.report/'design.json').exists():
        if json.loads((args.report/'design.json').read_text()) != design:
            raise ValueError('Existing experiment design differs')
    else:
        write_json(args.report/'design.json',design)
    cfg = config['reality_memory']
    encoder = RealityEncoder(ROOT/cfg['encoder']['path'],cfg['projector']['memory_dim'],
                             cfg['projector']['num_memory_tokens'],cfg['encoder']['image_size']).to(device).eval().requires_grad_(False)
    if encoder.identity != cache.identity:
        raise ValueError('Prefix encoder and reference cache identities differ')
    global_async = load_global_constant(config,cache,device,'async')
    pipeline = load_pipeline(config,device)
    torch.manual_seed(config['seed'])
    memory = make_memory(config).to(device).train()
    initial_digest = state_digest(memory)
    initial_params = {n:p.detach().cpu().clone() for n,p in memory.named_parameters()}
    opt = torch.optim.AdamW(memory.parameters(),lr=config['train']['lr'],weight_decay=config['train']['weight_decay'])
    warmup = config['train']['warmup_steps']
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1.,(step+1)/max(1,warmup)))
    batch_step, updates, records = 0,0,[]
    if saved is not None:
        if saved['initial_adapter_sha256'] != initial_digest:
            raise ValueError('Fresh adapter initialization changed')
        memory.load_state_dict(saved['memory'],strict=True)
        opt.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        batch_step,updates,records = saved['batch_step'],saved['optimizer_steps'],saved['records']
        torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state(saved['cuda_rng'],device)
    prepared = PreparedInputs(ds,encoder,pipeline,config,device)
    def checkpoint():
        path = args.output/f'update_{updates:04d}.pt'
        temporary = path.with_suffix('.tmp')
        torch.save({'stage':'r1a_matched','signature':signature,'initial_adapter_sha256':initial_digest,
                    'memory':memory.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),
                    'batch_step':batch_step,'optimizer_steps':updates,'records':records,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(device)},temporary)
        temporary.replace(path)
        return path
    idle=0
    while updates < args.max_updates:
        entry = schedule_entry(ds.rows,batch_step,config['seed'],config['train']['no_memory_probability'])
        item = prepared.get(entry['index'])
        opt.zero_grad(set_to_none=True)
        before = state_digest(memory) if entry['no_memory'] else None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,stats = r1_loss(pipeline,memory,item,args.branch,global_async,entry['noise_seed'],
                                 cfg['regularization']['delta_weight'],entry['no_memory'])
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite R1 loss')
        loss.backward()
        if any(p.requires_grad or p.grad is not None for m in (pipeline,encoder) for p in m.parameters()):
            raise RuntimeError('Backbone/DINO must remain frozen')
        lr = opt.param_groups[0]['lr']
        updated,norms,norm = optimizer_update(memory,opt,scheduler,entry['no_memory'],config['train']['grad_clip'])
        if entry['no_memory'] and before != state_digest(memory):
            raise RuntimeError('No-Memory batch changed adapter parameters')
        updates += int(updated)
        batch_step += 1
        idle = 0 if updated else idle+1
        if idle >= 100:
            raise RuntimeError('Unreachable effective update budget')
        record = {'batch_step':batch_step,'optimizer_steps':updates,'updated':updated,
                  **entry,'sample_id':item['sample_id'],'donor_sample_id':item['donor_sample_id'],
                  'loss':loss.item(),'video_loss':stats['video_loss'].item(),'delta_square':stats['delta_square'].item(),
                  'learning_rate':lr,'gradient_norm':norm,'parameter_gradient_norms':norms,
                  'selected_indices':stats['routing'].get('selected_indices',torch.empty(0,dtype=torch.long)).cpu().tolist(),
                  'no_memory_unchanged':bool(entry['no_memory'])}
        records.append(record)
        print(json.dumps({k:record[k] for k in ['batch_step','optimizer_steps','no_memory','video_loss','gradient_norm']}),flush=True)
        del loss,stats
        if updated and (updates == 2 or updates == args.max_updates):
            if updates == 2 and not any(v>0 for k,v in norms.items() if k.startswith('projector.')):
                raise RuntimeError('Second effective update must reach the projector')
            checkpoint()
    changes={n:(p.detach().cpu()-initial_params[n]).norm().item() for n,p in memory.named_parameters()}
    if not all(torch.isfinite(p).all() for p in memory.parameters()):
        raise RuntimeError('Nonfinite trained adapter')
    report={'status':'passed','stage':'r1a_matched','branch':args.branch,'optimizer_steps':updates,'batch_steps':batch_step,
            'initial_adapter_sha256':initial_digest,'final_adapter_sha256':state_digest(memory),'records':records,
            'parameter_change_norms':changes,'signature':signature,'backbones_frozen':True,
            'peak_vram_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),
            'checkpoint':str((args.output/f'update_{updates:04d}.pt').resolve())}
    write_json(args.report/f'train_{updates:04d}.json',report)
    if args.evaluate_dev:
        dev_inputs=PreparedInputs(dev,encoder,pipeline,config,device)
        evaluate(pipeline,memory,dev_inputs,global_async,config,args.report/f'dev_{updates:04d}.json')
    print(f'PASS {args.branch} {updates} effective updates',flush=True)


if __name__ == '__main__':
    main()
