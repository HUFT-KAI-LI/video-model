"""Inference-only history attention gate used by the release sweep."""
from contextlib import contextmanager


def _attention_modules(pipeline):
    root = getattr(pipeline, "generator", pipeline)
    return [module for module in root.modules()
            if module.__class__.__name__ == "CausalWanSelfAttention"]


_STATE_NAMES = ("history_gate", "history_component_gates", "history_path_gates",
                "history_layer_release")


def _capture_state(modules, missing):
    return [tuple(getattr(module, name, missing) for name in _STATE_NAMES)
            for module in modules]


def _restore_state(modules, states, missing):
    for module, values in zip(modules, states):
        for name, value in zip(_STATE_NAMES, values):
            if value is missing:
                delattr(module, name)
            else:
                setattr(module, name, value)


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
            module.history_component_gates = None
            module.history_path_gates = None
            module.history_layer_release = None
            count += 1
    if count == 0:
        raise RuntimeError("no CausalWanSelfAttention modules found")
    return count


@contextmanager
def history_gate(pipeline, gate: float):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    missing = object()
    old = _capture_state(modules, missing)
    set_history_gate(pipeline, gate)
    try:
        yield
    finally:
        _restore_state(modules, old, missing)


def set_history_component_gates(pipeline, gates) -> int:
    """Enable independent sink/old/recent gates on every self-attention module."""
    names = ("sink", "old", "recent")
    if set(gates) != set(names):
        raise ValueError("history component gates must define sink, old, and recent")
    values = {name: float(gates[name]) for name in names}
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("history component gates must be in [0, 1]")
    count = 0
    root = getattr(pipeline, "generator", pipeline)
    for module in root.modules():
        if module.__class__.__name__ == "CausalWanSelfAttention":
            module.history_gate = 1.0
            module.history_component_gates = dict(values)
            module.history_path_gates = None
            module.history_layer_release = None
            count += 1
    if count == 0:
        raise RuntimeError("no CausalWanSelfAttention modules found")
    return count


@contextmanager
def history_component_gates(pipeline, gates):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    missing = object()
    old = _capture_state(modules, missing)
    set_history_component_gates(pipeline, gates)
    try:
        yield
    finally:
        _restore_state(modules, old, missing)


def set_history_path_gates(pipeline, score, value) -> int:
    """Enable independent history score/access and value/content gates."""
    gates = {"score": float(score), "value": float(value)}
    if any(not 0.0 <= gate <= 1.0 for gate in gates.values()):
        raise ValueError("history path gates must be in [0, 1]")
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    if not modules:
        raise RuntimeError("no CausalWanSelfAttention modules found")
    for module in modules:
        module.history_gate = 1.0
        module.history_component_gates = None
        module.history_path_gates = dict(gates)
        module.history_layer_release = None
    return len(modules)


@contextmanager
def history_path_gates(pipeline, score, value):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    missing = object()
    old = _capture_state(modules, missing)
    set_history_path_gates(pipeline, score, value)
    try:
        yield
    finally:
        _restore_state(modules, old, missing)


def set_history_layer_release(pipeline, releases) -> int:
    """Install one full-to-current release coefficient per attention layer."""
    modules = _attention_modules(pipeline)
    values = [float(value) for value in releases]
    if not modules:
        raise RuntimeError("no CausalWanSelfAttention modules found")
    if len(values) != len(modules):
        raise ValueError(f"expected {len(modules)} layer releases, got {len(values)}")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("history layer releases must be in [0, 1]")
    for module, value in zip(modules, values):
        module.history_gate = 1.0
        module.history_component_gates = None
        module.history_path_gates = None
        module.history_layer_release = value
    return len(modules)


@contextmanager
def history_layer_release(pipeline, releases):
    modules = _attention_modules(pipeline)
    missing = object()
    old = _capture_state(modules, missing)
    set_history_layer_release(pipeline, releases)
    try:
        yield
    finally:
        _restore_state(modules, old, missing)
