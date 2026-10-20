"""Native text-to-video evaluation. No Teacher, Mask, Grounding or projector is loaded."""
import argparse
import csv
import gc
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from .data import caption_key, load_split
from .io_utils import sha256, write_json
from .runtime import ROOT, config, forward_fn, load_dit, load_text, load_trainable, load_vae, seed_everything, upstream


@torch.no_grad()
def generate(cfg, runs, output, limit=None, seed_count=None):
    upstream()
    from diffsynth.diffusion.flow_match import FlowMatchScheduler
    output = Path(output)
    if output.exists():
        raise FileExistsError('Refusing to overwrite generated results')
    output.mkdir(parents=True)
    seed_everything(cfg['seed'])
    prompts = json.loads((ROOT / 'compositional_prompts.json').read_text())
    if limit:
        prompts = prompts[:limit]
    training_captions = {r['caption_group'] for r in load_split(cfg['manifests'], 'train')}
    if any(caption_key(p['prompt']) in training_captions for p in prompts):
        raise ValueError('Evaluation prompt duplicates a training caption')
    device = cfg['devices']['evaluation']
    encoder, tokenizer = load_text(cfg, device)
    contexts = {}
    for text in [''] + [p['prompt'] for p in prompts]:
        ids, valid = tokenizer(text, return_mask=True, add_special_tokens=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            context = encoder(ids.to(device), valid.to(device))
        contexts[text] = (context * valid.to(device)[..., None]).cpu()
    del encoder, tokenizer, context
    gc.collect()
    torch.cuda.empty_cache()
    dit = load_dit(cfg, device).eval()
    vae = load_vae(cfg, device)
    native_forward = forward_fn()
    torch.cuda.reset_peak_memory_stats(device)
    reports, review_rows = [], []
    seeds = cfg['generation']['seeds'][:seed_count] if seed_count else cfg['generation']['seeds']
    first_videos = {}
    for group, path in runs.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['group'] != group:
            raise ValueError('Checkpoint arm label mismatch')
        if checkpoint['split_lock_sha256'] != sha256(Path(cfg['manifests']) / 'split_lock.json'):
            raise ValueError('Checkpoint was trained on a different split')
        load_trainable(dit, checkpoint['lora'])
        for prompt in prompts:
            for seed in seeds:
                start = time.monotonic()
                generator = torch.Generator('cpu').manual_seed(seed)
                shape = (1, 16, (cfg['frames'] - 1) // 4 + 1, cfg['height'] // 8, cfg['width'] // 8)
                latent = torch.randn(shape, generator=generator).to(device=device, dtype=torch.bfloat16)
                positive = contexts[prompt['prompt']].to(device)
                negative = contexts[''].to(device)
                scheduler = FlowMatchScheduler('Wan')
                scheduler.set_timesteps(cfg['generation']['steps'], shift=cfg['generation']['shift'])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    for t in scheduler.timesteps:
                        timestep = t[None].to(device=device, dtype=torch.bfloat16)
                        pred_pos = native_forward(dit=dit, latents=latent, timestep=timestep, context=positive)
                        pred_neg = native_forward(dit=dit, latents=latent, timestep=timestep, context=negative)
                        pred = pred_neg + cfg['generation']['cfg'] * (pred_pos - pred_neg)
                        latent = scheduler.step(pred, t, latent)
                    video = vae.decode(latent, device=device)
                frames = ((video[0].float().cpu().permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).byte().numpy()
                path = output / f'{group}_{prompt["id"]}_{seed}.mp4'
                imageio.mimwrite(path, frames, fps=cfg['fps'], codec='libx264', quality=8)
                report = dict(group=group, prompt=prompt, seed=seed, path=str(path.resolve()),
                              seconds=time.monotonic() - start, pixel_std=float(frames.std()),
                              frame_change=float(np.abs(np.diff(frames.astype(float), axis=0)).mean()),
                              train_steps=checkpoint['steps'], generation=cfg['generation'],
                              inference_modules=['umt5', 'Wan DiT + LoRA', 'native flow scheduler', 'Wan VAE'],
                              teacher_used=False, auxiliary_modules_used=False)
                reports.append(report)
                review_rows.append(dict(group=group, prompt_id=prompt['id'], seed=seed, path=str(path),
                                        attribute_correct='', action_correct='', binding_correct='',
                                        quality_regression='', generation_failure='', notes=''))
                if prompt is prompts[0] and seed == seeds[0]:
                    first_videos[group] = frames
                print(f'{group} {prompt["id"]} seed={seed} {report["seconds"]:.2f}s', flush=True)
    if set(first_videos) == {'A', 'B', 'C'}:
        joined = np.concatenate([first_videos[g] for g in ('A', 'B', 'C')], axis=2)
        imageio.mimwrite(output / 'side_by_side_A_B_C.mp4', joined, fps=cfg['fps'], codec='libx264', quality=8)
        sheet = np.concatenate([joined[i] for i in (0, len(joined) // 2, len(joined) - 1)], axis=0)
        imageio.imwrite(output / 'contact_sheet.png', sheet)
    with (output / 'human_review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    write_json(output / 'generation_report.json', dict(videos=reports,
               peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
               peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30,
               metric_status='VBench/T2V-CompBench not run; human review fields intentionally unscored',
               efficacy_conclusion='No efficacy claim from smoke checkpoints'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--A', required=True)
    parser.add_argument('--B', required=True)
    parser.add_argument('--C', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed-count', type=int)
    args = parser.parse_args()
    generate(config(args.config), {g: getattr(args, g) for g in ('A', 'B', 'C')}, args.output,
             args.limit, args.seed_count)
