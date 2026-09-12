# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

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
    'path_gated_attention',
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


def path_gated_attention(q, k_history, v_history, k_current, v_current,
                         score_gate, value_gate, **kwargs):
    """Attenuate history attention odds and value content independently.

    For 0 < score_gate < 1, log(score_gate) is added to every history
    logit before the joint history/current softmax. value_gate scales only
    history values. Exact native paths are retained at score endpoints.
    """
    alpha, beta = float(score_gate), float(value_gate)
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError("history score and value gates must be in [0,1]")
    if k_history.shape[1] != v_history.shape[1]:
        raise ValueError("history key/value lengths differ")
    if k_history.shape[1] == 0 or alpha == 0:
        return attention(q, k_current, v_current, **kwargs)

    keys = torch.cat([k_history, k_current], dim=1)
    values = torch.cat([v_history * beta, v_current], dim=1)
    if alpha == 1:
        return attention(q, keys, values, **kwargs)
    if kwargs.get("causal", False) or kwargs.get("dropout_p", 0.0) != 0:
        raise ValueError("fractional history-score gating is inference-only and non-causal")
    if kwargs.get("window_size", (-1, -1)) != (-1, -1):
        raise ValueError("path-gated attention expects a pre-sliced native window")

    if FLASH_ATTN_2_AVAILABLE and q.device.type == "cuda":
        q_lens = kwargs.get("q_lens")
        k_lens = kwargs.get("k_lens")
        if q_lens is not None or k_lens is not None:
            raise ValueError("path-gated attention does not support padded sequences")

        def native_with_lse(native_k, native_v):
            dtype = kwargs.get("dtype", torch.bfloat16)
            half_dtypes = (torch.float16, torch.bfloat16)
            q_native = q if q.dtype in half_dtypes else q.to(dtype)
            k_native = native_k if native_k.dtype in half_dtypes else native_k.to(dtype)
            v_native = native_v if native_v.dtype in half_dtypes else native_v.to(dtype)
            q_native = q_native.to(v_native.dtype)
            k_native = k_native.to(v_native.dtype)
            if kwargs.get("q_scale") is not None:
                q_native = q_native * kwargs["q_scale"]
            batch, q_length, heads = q_native.shape[:3]
            key_length = k_native.shape[1]
            q_flat = q_native.flatten(0, 1)
            k_flat = k_native.flatten(0, 1)
            v_flat = v_native.flatten(0, 1)
            q_cu = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * q_length
            k_cu = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * key_length
            output, lse, _ = flash_attn.flash_attn_varlen_func(
                q=q_flat, k=k_flat, v=v_flat, cu_seqlens_q=q_cu, cu_seqlens_k=k_cu,
                max_seqlen_q=q_length, max_seqlen_k=key_length,
                dropout_p=kwargs.get("dropout_p", 0.0),
                softmax_scale=kwargs.get("softmax_scale"), causal=kwargs.get("causal", False),
                window_size=kwargs.get("window_size", (-1, -1)),
                deterministic=kwargs.get("deterministic", False), return_attn_probs=True)
            lse = lse.transpose(0, 1).reshape(batch, q_length, heads, 1)
            return output.unflatten(0, (batch, q_length)).type_as(q), lse

        history_output, history_lse = native_with_lse(k_history, v_history * beta)
        current_output, current_lse = native_with_lse(k_current, v_current)
        history_weight = torch.sigmoid(history_lse + math.log(alpha) - current_lse)
        return (current_output.float() + history_weight *
                (history_output.float() - current_output.float())).type_as(q)

    q_lens = kwargs.pop("q_lens", None)
    k_lens = kwargs.pop("k_lens", None)
    if q_lens is not None or k_lens is not None:
        raise ValueError("path-gated attention does not support padded sequences")
    window_size = kwargs.pop("window_size", (-1, -1))
    if window_size != (-1, -1):
        raise ValueError("path-gated attention expects a pre-sliced native window")
    dtype = kwargs.pop("dtype", torch.bfloat16)
    q_scale = kwargs.pop("q_scale", None)
    softmax_scale = kwargs.pop("softmax_scale", None)
    causal = kwargs.pop("causal", False)
    dropout_p = kwargs.pop("dropout_p", 0.0)
    kwargs.pop("deterministic", None)
    kwargs.pop("fa_version", None)
    if kwargs:
        raise TypeError(f"unsupported path attention arguments: {sorted(kwargs)}")

    q_sdpa = q.transpose(1, 2).to(dtype)
    k_sdpa = keys.transpose(1, 2).to(dtype)
    v_sdpa = values.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q_sdpa = q_sdpa * q_scale
    bias = torch.zeros((1, 1, 1, keys.shape[1]), device=q.device, dtype=q_sdpa.dtype)
    bias[..., :k_history.shape[1]] = torch.log(
        torch.tensor(alpha, device=q.device, dtype=q_sdpa.dtype))
    output = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, attn_mask=bias, dropout_p=dropout_p,
        is_causal=causal, scale=softmax_scale)
    return output.transpose(1, 2).contiguous().type_as(q)
