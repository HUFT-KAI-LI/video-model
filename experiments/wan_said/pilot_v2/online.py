import time
from pathlib import Path

import torch

from ..alignment import masked_pool
from ..data import load_split
from ..io_utils import sha256
from ..runtime import load_text, load_vae
from .config import signature
from .teacher import load_teacher
from .video import sample_video, teacher_input


class OnlineFeatures:
    """Frozen encoders; only the current batch lives in memory, never a disk feature cache."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.text = self.tokenizer = self.vae = self.teacher = None
        self.calls = []
        self.allowed = {r['sample_id']: r for s in ('train', 'val') for r in load_split(cfg['manifests'], s)}
        self.preprocessing_hash = signature(cfg)

    @torch.no_grad()
    def __call__(self, row, kinds=('text', 'teacher', 'vae')):
        if row['sample_id'] not in self.allowed or row != self.allowed[row['sample_id']]:
            raise ValueError('Online training features accept only the frozen train/validation manifests')
        if row['preprocessing_hash'] != self.preprocessing_hash:
            raise ValueError('Manifest preprocessing differs from the active ViT-Base protocol')
        cfg, devices = self.cfg, self.cfg['devices']
        result = {}
        start = time.monotonic()
        if 'text' in kinds:
            if self.text is None:
                self.text, self.tokenizer = load_text(cfg, devices['text'])
            ids, valid = self.tokenizer(row['caption'], return_mask=True, add_special_tokens=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hidden = self.text(ids.to(devices['text']), valid.to(devices['text']))
            result['text'] = dict(context=(hidden*valid.to(hidden.device)[..., None]).cpu(),
                                  pooled=masked_pool(hidden, valid.to(hidden.device)).float().cpu())
        if set(kinds) & {'teacher', 'vae'}:
            images, sampling = sample_video(row, cfg)
            result['sampling'] = sampling
            if 'teacher' in kinds:
                if self.teacher is None:
                    self.teacher = load_teacher(cfg, devices['teacher'])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    # raw_tokens stay available to future modules without being written to disk.
                    result['teacher'] = {k: v.cpu() for k, v in self.teacher(
                        teacher_input(images, cfg).to(devices['teacher'])).items()}
            if 'vae' in kinds:
                if self.vae is None:
                    self.vae = load_vae(cfg, devices['vae'])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    latent = self.vae.encode((images*2-1).permute(1, 0, 2, 3)[None].to(torch.bfloat16),
                                             device=devices['vae'], tiled=cfg['vae_tiled'])
                expected = (1, 16, (cfg['frames']-1)//4+1, cfg['height']//8, cfg['width']//8)
                if tuple(latent.shape) != expected:
                    raise ValueError(f'Unexpected Wan VAE latent shape: {tuple(latent.shape)} != {expected}')
                result['vae'] = dict(latents=latent.cpu())
        for payload in result.values():
            if isinstance(payload, dict):
                for value in payload.values():
                    if torch.is_tensor(value) and not torch.isfinite(value).all():
                        raise FloatingPointError('Nonfinite online encoder feature')
        for encoder in (self.text, self.vae, self.teacher):
            if encoder is not None and any(p.requires_grad or p.grad is not None for p in encoder.parameters()):
                raise AssertionError('Frozen encoder received gradients')
        self.calls.append(dict(sample_id=row['sample_id'], split=row['split'], kinds=list(kinds),
                               seconds=time.monotonic()-start))
        return result


def experiment_identity(cfg):
    return dict(preprocessing_hash=signature(cfg),
                split_lock_sha256=sha256(Path(cfg['manifests'])/'split_lock.json'),
                wan_weights={name: sha256(Path(cfg['wan_weights'])/name) for name in
                             ('diffusion_pytorch_model.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth')},
                lora={key: cfg[key] for key in ('seed', 'lora_rank', 'lora_alpha', 'lora_targets')},
                training={key: cfg[key] for key in ('dim', 'lr', 'ground_lr', 'ground_batch',
                          'sparse_weight', 'distill_weight', 'distill_max_sigma',
                          'gradient_checkpointing', 'gradient_clip', 'weight_decay', 'deterministic_attention')},
                shared_algorithm_sha256={name: sha256(Path(__file__).resolve().parents[1]/name)
                                         for name in ('alignment.py', 'runtime.py', 'train.py')})
