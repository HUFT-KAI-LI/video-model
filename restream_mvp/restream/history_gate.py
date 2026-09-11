"""Inference-only history attention gate used by the release sweep."""
from contextlib import contextmanager


def set_history_gate(pipeline, gate: float) -> int:
    """Set gate on every streaming self-attention module; return count."""
    value = float(gate)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"history gate must be in [0, 1], got {gate}")
    count = 0
    root = getattr(pipeline, "generator", pipeline)
    for module in root.modules():
        if module.__class__.__name__ == "CausalWanSelfAttention":
            module.history_gate = value
            count += 1
    if count == 0:
        raise RuntimeError("no CausalWanSelfAttention modules found")
    return count


@contextmanager
def history_gate(pipeline, gate: float):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    missing = object()
    old = [getattr(m, "history_gate", missing) for m in modules]
    set_history_gate(pipeline, gate)
    try:
        yield
    finally:
        for module, value in zip(modules, old):
            if value is missing:
                delattr(module, "history_gate")
            else:
                module.history_gate = value
