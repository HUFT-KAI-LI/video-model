import hashlib
import json
from pathlib import Path

from ..runtime import ROOT
from ..io_utils import sha256

HERE = Path(__file__).resolve().parent
SOURCE_LOCK = json.loads((ROOT / 'protocol_v2/sources_lock.json').read_text())
PREPROCESS_KEYS = ('protocol_version', 'frames', 'height', 'width', 'teacher_model',
                   'teacher_dim', 'teacher_height', 'teacher_width',
                   'teacher_position_embedding', 'temporal_sampling_policy',
                   'spatial_crop_policy', 'normalization_mean', 'normalization_std', 'vae_tiled')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate(cfg):
    if cfg['frames'] < 5 or (cfg['frames'] - 1) % 4:
        raise ValueError('Wan requires frames=4k+1; the teacher receives frames-1')
    if any(cfg[k] <= 0 or cfg[k] % 16 for k in ('height', 'width', 'teacher_height', 'teacher_width')):
        raise ValueError('Spatial dimensions must be positive multiples of 16')
    if cfg['height'] * cfg['teacher_width'] != cfg['width'] * cfg['teacher_height']:
        raise ValueError('Teacher and Wan must have the same aspect ratio')
    if cfg['teacher_dim'] != 768 or cfg['teacher_model'] != 'videomaev2_vit_base':
        raise ValueError('Pilot v2 requires the actual ViT-Base, not ViT-Small')
    if cfg['feature_cache_enabled']:
        raise ValueError('The pilot is online-only; full feature cache writes are disabled')
    return cfg


def load_config(path=None):
    cfg = json.loads(Path(path or HERE / 'config.json').read_text())
    for key in ('wan_weights', 'teacher_weights', 'parent_manifests', 'manifests'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return validate(cfg)


def preprocessing(cfg):
    return dict(**{k: cfg[k] for k in PREPROCESS_KEYS},
                teacher_checkpoint_sha256=SOURCE_LOCK['teacher']['sha256'],
                teacher_source_commit=SOURCE_LOCK['VideoREPA']['commit'],
                teacher_patch_size=16, teacher_tubelet_size=2,
                implementation_sha256={name: sha256(HERE/name) for name in ('video.py', 'teacher.py')})


def signature(cfg):
    return digest(preprocessing(cfg))
