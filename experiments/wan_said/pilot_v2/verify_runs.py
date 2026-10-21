"""Bounded real-model resume checks against an uninterrupted deterministic run."""
import argparse
import gc
from pathlib import Path

import torch

from ..io_utils import sha256, write_json
from .config import load_config
from .grounding import train as train_grounding
from .online import OnlineFeatures
from .train import train


def verify(cfg, smoke_root, output):
    smoke_root, output = Path(smoke_root), Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh verification output')
    output.mkdir(parents=True)
    before = {g: sha256(smoke_root/g/'checkpoint.pt') for g in ('A', 'B', 'C')}
    provider = OnlineFeatures(cfg)
    resumed = train(cfg, 'A', output/'A_resumed', 3, resume=smoke_root/'A/checkpoint.pt',
                    smoke=True, provider=provider)
    expected_hash = resumed['final_lora_sha256']
    del resumed
    gc.collect()
    torch.cuda.empty_cache()
    whole = train(cfg, 'A', output/'A_uninterrupted', 3, smoke=True, provider=provider)
    exact = whole['final_lora_sha256'] == expected_hash
    del whole
    gc.collect()
    torch.cuda.empty_cache()
    if not exact:
        raise AssertionError('Resumed LoRA state differs from uninterrupted training')
    said = train(cfg, 'C', output/'C_resumed', 3, smoke_root/'ground_said/grounding.pt',
                 resume=smoke_root/'C/checkpoint.pt', smoke=True, provider=provider)
    if said['steps'] != 3 or len(said['optimizer']['state']) == 0:
        raise AssertionError('C adapter/optimizer did not resume')
    del said
    gc.collect()
    torch.cuda.empty_cache()
    ground = train_grounding(cfg, 'said', output/'ground_said_resumed', 4,
                             train_limit=8, val_limit=4, resume=smoke_root/'ground_said/grounding.pt',
                             provider=provider)
    if ground['steps'] != 4 or len(ground['optimizer']['state']) == 0:
        raise AssertionError('Grounding/optimizer did not resume')
    if before != {g: sha256(smoke_root/g/'checkpoint.pt') for g in before}:
        raise AssertionError('Resume overwrote an original checkpoint')
    result = dict(a_resume_matches_uninterrupted_exactly=exact, a_step3_lora_sha256=expected_hash,
                  c_resume_total_steps=3, grounding_resume_total_steps=4,
                  source_checkpoints_unchanged=True, optimizer_restored=True,
                  feature_cache_bytes=0, formal_efficacy_evidence=False)
    write_json(output/'verification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--smoke-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(load_config(args.config), args.smoke_root, args.output)
