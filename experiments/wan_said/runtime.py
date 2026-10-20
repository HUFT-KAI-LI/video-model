import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def config(path=None):
    path = Path(path) if path else ROOT / 'config.json'
    cfg = json.loads(path.read_text())
    for key in ('teacher_weights', 'manifests', 'cache'):
        cfg[key] = str((ROOT / cfg[key]).resolve())
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def upstream():
    lock = json.loads((ROOT / 'upstream_lock.json').read_text())
    for name in ('DiffSynth-Studio', 'VideoMAEv2'):
        path = ROOT / 'vendor' / name
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != lock[name]['commit']:
            raise RuntimeError(f'Upstream revision mismatch for {name}: {revision}')
        if subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError(f'Upstream has local modifications: {name}')
    path = str(ROOT / 'vendor' / 'DiffSynth-Studio')
    if path not in sys.path:
        sys.path.insert(0, path)


def load_dit(cfg, device, lora=True):
    upstream()
    from diffsynth.models.wan_video_dit import WanModel, precompute_freqs_cis_3d
    from diffsynth.utils.state_dict_converters.wan_video_dit import WanVideoDiTStateDictConverter
    from safetensors.torch import load_file

    with torch.device('meta'):
        model = WanModel(dim=1536, in_dim=16, ffn_dim=8960, out_dim=16, text_dim=4096,
                         freq_dim=256, eps=1e-6, patch_size=(1, 2, 2), num_heads=12,
                         num_layers=30, has_image_input=False)
    weights = load_file(str(Path(cfg['wan_weights']) / 'diffusion_pytorch_model.safetensors'))
    model.load_state_dict(WanVideoDiTStateDictConverter(weights), strict=True, assign=True)
    del weights
    model.freqs = precompute_freqs_cis_3d(128)
    model.requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    if lora:
        from peft import LoraConfig, inject_adapter_in_model
        model = inject_adapter_in_model(LoraConfig(r=cfg['lora_rank'], lora_alpha=cfg['lora_alpha'],
                                                  target_modules=cfg['lora_targets'], lora_dropout=0.0), model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'lora_' not in name:
                    raise RuntimeError(f'Unexpected trainable backbone parameter: {name}')
                param.data = param.data.float()
    return model


def load_text(cfg, device):
    upstream()
    from diffsynth.models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
    with torch.device('meta'):
        model = WanTextEncoder()
    weights = torch.load(Path(cfg['wan_weights']) / 'models_t5_umt5-xxl-enc-bf16.pth',
                         map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(weights, strict=True, assign=True)
    del weights
    model = model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    tokenizer = HuggingfaceTokenizer(str(Path(cfg['wan_weights']) / 'google/umt5-xxl'),
                                    seq_len=512, clean='whitespace', local_files_only=True)
    return model, tokenizer


def load_vae(cfg, device):
    upstream()
    from diffsynth.models.wan_video_vae import WanVideoVAE
    from diffsynth.utils.state_dict_converters.wan_video_vae import WanVideoVAEStateDictConverter
    model = WanVideoVAE()
    weights = torch.load(Path(cfg['wan_weights']) / 'Wan2.1_VAE.pth', map_location='cpu',
                         weights_only=True, mmap=True)
    model.load_state_dict(WanVideoVAEStateDictConverter(weights), strict=True, assign=True)
    del weights
    return model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)


def forward_fn():
    upstream()
    from diffsynth.pipelines.wan_video import model_fn_wan_video
    return model_fn_wan_video


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def trainable_state(model):
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def load_trainable(model, state):
    parameters = dict(model.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if set(state) != expected:
        raise ValueError(f'Adapter keys mismatch; missing={expected - set(state)}, extra={set(state) - expected}')
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))
