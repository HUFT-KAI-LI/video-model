"""Scope deterministic kernel settings to the new experiment, leaving native math intact."""
import os
from contextlib import contextmanager
from functools import partial, wraps
from types import SimpleNamespace

import torch

from ..runtime import upstream

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


@contextmanager
def deterministic_attention(enabled=True):
    if not enabled:
        raise ValueError('Pilot v2 requires deterministic training kernels for A/B/C and resume checks')
    upstream()
    from diffsynth.models import wan_video_dit
    if wan_video_dit.FLASH_ATTN_3_AVAILABLE or wan_video_dit.SAGE_ATTN_AVAILABLE:
        raise ValueError('This execution protocol was validated with FlashAttention 2 or native PyTorch SDPA only')
    previous = wan_video_dit.flash_attn if wan_video_dit.FLASH_ATTN_2_AVAILABLE else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        if previous is not None:
            # Keep the upstream forward and attention operator; only select its documented backward kernel.
            wan_video_dit.flash_attn = SimpleNamespace(
                flash_attn_func=partial(previous.flash_attn_func, deterministic=True))
        yield
    finally:
        if previous is not None:
            wan_video_dit.flash_attn = previous
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def deterministic_training(function):
    @wraps(function)
    def wrapped(cfg, *args, **kwargs):
        with deterministic_attention(cfg['deterministic_attention']):
            return function(cfg, *args, **kwargs)
    return wrapped
