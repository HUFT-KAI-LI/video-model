# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch

try:
    import flash_attn_interface

    def is_hopper_gpu():
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            return major >= 9  # Hopper Compute Capability == 9.0
        return False
    FLASH_ATTN_3_AVAILABLE = is_hopper_gpu()
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

# FLASH_ATTN_3_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
    'gated_attention',
    'component_gated_attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out

def gated_attention(q, k_history, v_history, k_current, v_current, gate, **kwargs):
    """Interpolate two native attention outputs; gate is contribution strength."""
    g = float(gate)
    if not 0 <= g <= 1: raise ValueError("history gate must be in [0,1]")
    if g == 1 or k_history.shape[1] == 0:
        return attention(q, torch.cat([k_history, k_current], 1),
                         torch.cat([v_history, v_current], 1), **kwargs)
    current = attention(q, k_current, v_current, **kwargs)
    if g == 0: return current
    full = attention(q, torch.cat([k_history, k_current], 1),
                     torch.cat([v_history, v_current], 1), **kwargs)
    return current + g * (full - current)


def component_gated_attention(q, components, k_current, v_current, gates, **kwargs):
    """Independently interpolate native attention over named history components.

    A gate is the inclusion weight for its component. Multiple fractional gates
    use the multilinear extension: native attention is evaluated for every
    inclusion subset and combined by the corresponding product weight.
    """
    names = ("sink", "old", "recent")
    if tuple(components) != names or set(gates) != set(names):
        raise ValueError("history components and gates must be sink, old, recent in order")
    values = {name: float(gates[name]) for name in names}
    if any(not 0 <= value <= 1 for value in values.values()):
        raise ValueError("history component gates must be in [0,1]")
    if any(components[name][0].shape[1] != components[name][1].shape[1] for name in names):
        raise ValueError("history component key/value lengths differ")

    active = [name for name in names
              if components[name][0].shape[1] > 0 and values[name] not in (0.0, 1.0)]
    fixed = {name: values[name] == 1.0 for name in names}

    def native(included):
        keys = [components[name][0] for name in names if included[name]] + [k_current]
        vals = [components[name][1] for name in names if included[name]] + [v_current]
        return attention(q, torch.cat(keys, dim=1), torch.cat(vals, dim=1), **kwargs)

    if not active:
        return native(fixed)
    if len(active) == 1:
        name = active[0]
        without = dict(fixed)
        without[name] = False
        baseline = native(without)
        included = dict(fixed)
        included[name] = True
        full = native(included)
        return baseline + values[name] * (full - baseline)

    output = None
    for mask in range(1 << len(active)):
        included = dict(fixed)
        weight = 1.0
        for index, name in enumerate(active):
            present = bool(mask & (1 << index))
            included[name] = present
            weight *= values[name] if present else 1.0 - values[name]
        value = native(included)
        output = value * weight if output is None else output + value * weight
    return output
