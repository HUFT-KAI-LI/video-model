import importlib.util
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F

from ..io_utils import sha256
from ..runtime import ROOT
from .config import SOURCE_LOCK, digest, preprocessing, signature


def interpolate_positions(position, source_grid, target_grid):
    st, sh, sw = source_grid
    tt, th, tw = target_grid
    if position.ndim != 3 or position.shape[1] != st*sh*sw or min(target_grid) < 1:
        raise ValueError('Position embedding grid mismatch')
    b, _, c = position.shape
    value = position.float().reshape(b*st, sh, sw, c).permute(0, 3, 1, 2)
    value = F.interpolate(value, (th, tw), mode='bicubic', align_corners=False)
    value = value.permute(0, 2, 3, 1).reshape(b, st, th, tw, c).permute(0, 4, 1, 2, 3)
    value = F.interpolate(value, (tt, th, tw), mode='trilinear', align_corners=False)
    return value.permute(0, 2, 3, 4, 1).flatten(1, 3).contiguous()


def source_module():
    repo = ROOT/'vendor/VideoREPA'
    lock = SOURCE_LOCK['VideoREPA']
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != lock['commit']:
        raise ValueError('VideoREPA source revision mismatch')
    path = repo/'finetune/models/cogvideox_t2v_align/models/ssl/VideoMAEv2.py'
    if sha256(path) != lock['teacher_source_sha256']:
        raise ValueError('VideoREPA teacher source was modified')
    name = 'wan_said_pilot_v2_videomaev2'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class FrozenVideoTeacher(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('base_positions', model.pos_embed.detach().clone(), persistent=False)
        self.load_report = {}
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, video):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] % 2 or any(x % 16 for x in video.shape[-2:]):
            raise ValueError('Teacher expects Bx3x(even T)x(16h)x(16w)')
        grid = (video.shape[2]//2, video.shape[3]//16, video.shape[4]//16)
        self.model.pos_embed = interpolate_positions(self.base_positions, (8, 14, 14), grid)
        tokens = self.model(video)
        if tokens.shape != (video.shape[0], grid[0]*grid[1]*grid[2], 768) or not torch.isfinite(tokens).all():
            raise ValueError('Invalid ViT-Base teacher output')
        self.last_shapes = dict(input=list(video.shape), patch_embedding=list(tokens.shape),
                                position_embedding=list(self.model.pos_embed.shape), token_grid=list(grid))
        return dict(raw_tokens=tokens, pooled=tokens.float().mean(1))


def load_teacher(cfg, device):
    if sha256(cfg['teacher_weights']) != SOURCE_LOCK['teacher']['sha256']:
        raise ValueError('ViT-Base checkpoint SHA256 mismatch')
    model = source_module().vit_base_patch16_224(num_classes=710, all_frames=16)
    checkpoint = torch.load(cfg['teacher_weights'], map_location='cpu', weights_only=True, mmap=True)['module']
    ignored = {k: checkpoint.pop(k) for k in ('head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias')}
    if ignored['head.weight'].shape != (710, 768) or ignored['fc_norm.weight'].shape != (768,):
        raise ValueError('Unexpected classifier weights; not the expected ViT-Base checkpoint')
    result = model.load_state_dict(checkpoint, strict=True)
    teacher = FrozenVideoTeacher(model.to(device=device, dtype=torch.bfloat16)).to(device=device)
    teacher.load_report = dict(missing_keys=list(result.missing_keys), unexpected_keys=list(result.unexpected_keys),
                              explicitly_removed_classifier_keys=sorted(ignored),
                              pooling='mean of final block raw tokens; no classifier/fc_norm, matching VideoREPA',
                              checkpoint_sha256=SOURCE_LOCK['teacher']['sha256'])
    return teacher


def feature_metadata(cfg, row):
    return dict(**preprocessing(cfg), preprocessing_hash=signature(cfg), source_video_sha256=row['sha256'],
                source_caption_hash=digest(row['caption']),
                teacher_input_frames=cfg['frames']-1, teacher_input_height=cfg['teacher_height'],
                teacher_input_width=cfg['teacher_width'], teacher_architecture=cfg['teacher_model'],
                teacher_hidden_dim=768)


def validate_feature_metadata(metadata, cfg, row):
    if metadata != feature_metadata(cfg, row):
        raise ValueError('Rejecting stale feature identity, preprocessing, split sample, or ViT-Small cache')
