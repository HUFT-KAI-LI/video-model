#!/usr/bin/env python3
"""Read-only preflight; does not download models or start generation."""
import argparse
import importlib.util
import json
from pathlib import Path
from common import HERE, ROOT, build_plan, geometry, load_config, load_prompts, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=HERE/'generation_config.yaml')
    p.add_argument('--output',type=Path)
    p.add_argument('--verify-weights',action='store_true',help='Hash all assets; requires CUDA and installed runtime')
    args=p.parse_args()
    cfg=load_config(args.config)
    prompts=load_prompts(HERE/'prompts.jsonl')
    deps={m:importlib.util.find_spec(m) is not None for m in
          ('torch','torchvision','flash_attn','omegaconf','peft','transformers','av','numpy','PIL','matplotlib')}
    cuda=False
    if deps['torch']:
        import torch
        cuda=torch.cuda.is_available()
    wan=ROOT/cfg['model']['wan']
    longlive=ROOT/cfg['model']['longlive']
    assets={str(p):p.is_file() for p in (wan/'Wan2.1_VAE.pth',wan/'diffusion_pytorch_model.safetensors',
             wan/'models_t5_umt5-xxl-enc-bf16.pth',wan/'google/umt5-xxl/tokenizer.json',
             longlive/'models/longlive_base.pt',longlive/'models/lora.pt')}
    result=dict(planned_videos=len(build_plan(cfg,prompts)),geometry={k:v for k,v in geometry(cfg['duration_sec']).items() if k!='blocks'},
                dependencies=deps,assets=assets,cuda_available=cuda,
                ready_to_generate=cuda and all(assets.values()) and all(v for k,v in deps.items() if k!='matplotlib'),
                ready_to_plot=deps['matplotlib'],gpu_rollout_validated=False)
    if args.verify_weights:
        from backend import provenance
        result['provenance']=provenance(cfg)
    if args.output:
        write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
