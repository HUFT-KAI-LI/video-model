"""Real online A/B/C engineering checks, not a formal efficacy experiment."""
import argparse
import gc
from pathlib import Path

import torch

from ..io_utils import write_json
from .config import load_config
from .grounding import train as train_grounding
from .online import OnlineFeatures
from .train import train


def run(cfg, output, ground_steps=3, train_steps=2):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh smoke output')
    output.mkdir(parents=True)
    provider = OnlineFeatures(cfg)
    ground = {}
    for mode in ('full', 'said'):
        train_grounding(cfg, mode, output/f'ground_{mode}', ground_steps,
                        train_limit=8, val_limit=4, provider=provider)
        ground[mode] = output/f'ground_{mode}/grounding.pt'
    outputs = {}
    for group in ('A', 'B', 'C'):
        checkpoint = None if group == 'A' else ground['full' if group == 'B' else 'said']
        outputs[group] = train(cfg, group, output/group, train_steps, checkpoint,
                               smoke=True, gradient_audit=True, provider=provider)
        gc.collect()
        torch.cuda.empty_cache()
    initial = {value['initial_lora_sha256'] for value in outputs.values()}
    if len(initial) != 1:
        raise AssertionError('A/B/C LoRA initializations differ')
    for i in range(train_steps):
        reference = outputs['A']['records'][i]
        for group in ('B', 'C'):
            record = outputs[group]['records'][i]
            for field in ('sample_id', 'noise_seed', 'timestep', 'sigma', 'latent_sha256', 'sampling'):
                if reference[field] != record[field]:
                    raise AssertionError(f'A/{group} data/noise/preprocessing differs: {field}')
    result = dict(groups=list(outputs), real_model_steps_per_group=train_steps,
                  grounding_steps_per_mode=ground_steps, common_initial_lora_sha256=initial.pop(),
                  data_and_noise_identical=True, teacher_frozen=True,
                  full_and_said_share_teacher=True, feature_cache_bytes=0,
                  frames=cfg['frames'], height=cfg['height'], width=cfg['width'],
                  teacher_dim=768, teacher_input=[cfg['frames']-1, cfg['teacher_height'], cfg['teacher_width']],
                  online_calls=provider.calls, formal_efficacy_evidence=False)
    write_json(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--ground-steps', type=int, default=3)
    parser.add_argument('--train-steps', type=int, default=2)
    args = parser.parse_args()
    run(load_config(args.config), args.output, args.ground_steps, args.train_steps)
