import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import av
import numpy as np
import torch
from torch.nn import functional as F

from .alignment import masked_pool
from .data import load_split
from .io_utils import sha256, write_json
from .runtime import ROOT, config, load_text, load_vae, save_torch, seed_everything, upstream


def sample_video(row, cfg):
    """Use decoded PTS and nearest-previous frames, with a shared deterministic crop."""
    if sha256(row['video']) != row['sha256']:
        raise ValueError(f'Video content changed: {row["video"]}')
    times = row['window_start'] + np.arange(cfg['frames']) / cfg['fps']
    selected, actual = [], []
    previous, previous_time = None, None
    cursor = 0
    with av.open(row['video']) as container:
        for frame in container.decode(video=0):
            timestamp = float(frame.time)
            pixels = frame.to_ndarray(format='rgb24')
            while cursor < len(times) and times[cursor] < timestamp - 1e-7:
                selected.append(previous if previous is not None else pixels)
                actual.append(previous_time if previous_time is not None else timestamp)
                cursor += 1
            previous, previous_time = pixels, timestamp
            if cursor == len(times):
                break
        while cursor < len(times):
            if previous is None or times[cursor] > previous_time + .15:
                raise ValueError('Requested training window exceeds available video')
            selected.append(previous)
            actual.append(previous_time)
            cursor += 1
    images = torch.from_numpy(np.stack(selected)).permute(0, 3, 1, 2).float() / 255
    height, width = images.shape[-2:]
    ratio = max(cfg['height'] / height, cfg['width'] / width)
    images = F.interpolate(images, size=(round(height * ratio), round(width * ratio)),
                           mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    top = (images.shape[-2] - cfg['height']) // 2
    left = (images.shape[-1] - cfg['width']) // 2
    images = images[:, :, top:top + cfg['height'], left:left + cfg['width']]
    return images, dict(requested_seconds=times.tolist(), actual_seconds=actual,
                        policy='nearest_previous_pts', crop='resize_cover_center',
                        height=cfg['height'], width=cfg['width'], fps=cfg['fps'])


def load_teacher(cfg, device):
    upstream()
    lock = json.loads((ROOT / 'upstream_lock.json').read_text())['teacher']
    if sha256(cfg['teacher_weights']) != lock['sha256']:
        raise ValueError('Teacher weight checksum mismatch')
    path = ROOT / 'vendor/VideoMAEv2/models/modeling_finetune.py'
    spec = importlib.util.spec_from_file_location('wan_said_videomaev2', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    teacher = module.vit_small_patch16_224(num_classes=710, all_frames=16)
    state = torch.load(cfg['teacher_weights'], map_location='cpu', weights_only=True)['module']
    teacher.load_state_dict(state, strict=True)
    return teacher.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)


@torch.no_grad()
def cache(kind, cfg, device):
    seed_everything(cfg['seed'])
    rows = [row for split in ('train', 'val') for row in load_split(cfg['manifests'], split)]
    cache_root = Path(cfg['cache']) / kind
    cache_root.mkdir(parents=True, exist_ok=True)
    weight = {'text': Path(cfg['wan_weights']) / 'models_t5_umt5-xxl-enc-bf16.pth',
              'vae': Path(cfg['wan_weights']) / 'Wan2.1_VAE.pth', 'teacher': cfg['teacher_weights']}[kind]
    provenance = dict(kind=kind, weight_sha256=sha256(weight),
                      split_lock_sha256=sha256(Path(cfg['manifests']) / 'split_lock.json'),
                      preprocessing={k: cfg[k] for k in ('frames', 'fps', 'height', 'width')},
                      upstream=json.loads((ROOT / 'upstream_lock.json').read_text()))
    index = cache_root / 'provenance.json'
    if index.exists() and json.loads(index.read_text()) != provenance:
        raise ValueError(f'Cache provenance differs: {index}')
    write_json(index, provenance)
    start = time.monotonic()
    if kind == 'text':
        model, tokenizer = load_text(cfg, device)
    elif kind == 'vae':
        model = load_vae(cfg, device)
    else:
        model = load_teacher(cfg, device)
    assert not any(p.requires_grad for p in model.parameters())
    for i, row in enumerate(rows):
        path = cache_root / f'{row["sample_id"]}.pt'
        if path.exists():
            continue
        payload = dict(sample_id=row['sample_id'], split=row['split'],
                       manifest_sha256=sha256(Path(cfg['manifests']) / f'{row["split"]}.jsonl'))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if kind == 'text':
                ids, valid = tokenizer(row['caption'], return_mask=True, add_special_tokens=True)
                hidden = model(ids.to(device), valid.to(device))
                pooled = masked_pool(hidden, valid.to(device))
                hidden = hidden * valid.to(device)[..., None]
                payload.update(context=hidden.cpu(), valid=valid.bool(), pooled=pooled.cpu())
            else:
                images, sampling = sample_video(row, cfg)
                payload['sampling'] = sampling
                if kind == 'vae':
                    latents = model.encode((images * 2 - 1).permute(1, 0, 2, 3)[None].to(torch.bfloat16),
                                           device=device, tiled=False)
                    payload['latents'] = latents.cpu()
                else:
                    indices = np.linspace(0, cfg['frames'] - 1, 16).round().astype(int)
                    video = F.interpolate(images[indices], (224, 224), mode='bicubic',
                                          align_corners=False, antialias=True).clamp(0, 1)
                    video = (video - torch.tensor([.485, .456, .406])[None, :, None, None]) / \
                        torch.tensor([.229, .224, .225])[None, :, None, None]
                    tokens = []
                    handle = model.blocks[-1].register_forward_hook(lambda m, a, o: tokens.append(o.detach()))
                    try:
                        pooled = model.forward_features(video.permute(1, 0, 2, 3)[None].to(device))
                    finally:
                        handle.remove()
                    payload.update(pooled=pooled.float().cpu(), raw_tokens=tokens[0].cpu(),
                                   teacher_frame_indices=indices.tolist(),
                                   teacher_actual_seconds=[sampling['actual_seconds'][j] for j in indices])
        for value in payload.values():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite {kind} cache: {path}')
        save_torch(path, payload)
        print(f'{kind} {i + 1}/{len(rows)} {row["split"]} {row["sample_id"]} '
              f'elapsed={time.monotonic() - start:.1f}s', flush=True)
    write_json(cache_root / 'completion.json', dict(count=len(rows), seconds=time.monotonic() - start,
               device=device, peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2 ** 30,
               peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2 ** 30))


def load_features(cfg, row, kinds=('text', 'teacher', 'vae')):
    result = {}
    for kind in kinds:
        directory = Path(cfg['cache']) / kind
        provenance = json.loads((directory / 'provenance.json').read_text())
        if provenance['split_lock_sha256'] != sha256(Path(cfg['manifests']) / 'split_lock.json'):
            raise ValueError('Features belong to a different data split')
        if provenance['preprocessing'] != {k: cfg[k] for k in ('frames', 'fps', 'height', 'width')}:
            raise ValueError('Features were computed with different video preprocessing')
        if provenance['upstream'] != json.loads((ROOT / 'upstream_lock.json').read_text()):
            raise ValueError('Feature upstream provenance changed')
        payload = torch.load(directory / f'{row["sample_id"]}.pt', map_location='cpu', weights_only=True)
        if payload['split'] != row['split'] or payload['sample_id'] != row['sample_id']:
            raise ValueError('Feature sample/split mismatch')
        if payload['manifest_sha256'] != sha256(Path(cfg['manifests']) / f'{row["split"]}.jsonl'):
            raise ValueError('Features belong to a modified manifest')
        result[kind] = payload
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['text', 'vae', 'teacher'], required=True)
    parser.add_argument('--config')
    parser.add_argument('--device')
    args = parser.parse_args()
    cfg = config(args.config)
    cache(args.kind, cfg, args.device or cfg['devices'][f'{args.kind}_cache'])
