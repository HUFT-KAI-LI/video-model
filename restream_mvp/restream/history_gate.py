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
            module.history_component_gates = None
            module.history_path_gates = None
            count += 1
    if count == 0:
        raise RuntimeError("no CausalWanSelfAttention modules found")
    return count


@contextmanager
def history_gate(pipeline, gate: float):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    missing = object()
    old = [(getattr(m, "history_gate", missing),
            getattr(m, "history_component_gates", missing),
            getattr(m, "history_path_gates", missing)) for m in modules]
    set_history_gate(pipeline, gate)
    try:
        yield
    finally:
        for module, (gate_value, component_value, path_value) in zip(modules, old):
            if gate_value is missing:
                delattr(module, "history_gate")
            else:
                module.history_gate = gate_value
            if component_value is missing:
                delattr(module, "history_component_gates")
            else:
                module.history_component_gates = component_value
            if path_value is missing:
                delattr(module, "history_path_gates")
            else:
                module.history_path_gates = path_value


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
            count += 1
    if count == 0:
        raise RuntimeError("no CausalWanSelfAttention modules found")
    return count


@contextmanager
def history_component_gates(pipeline, gates):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    missing = object()
    old = [(getattr(m, "history_gate", missing),
            getattr(m, "history_component_gates", missing),
            getattr(m, "history_path_gates", missing)) for m in modules]
    set_history_component_gates(pipeline, gates)
    try:
        yield
    finally:
        for module, (gate_value, component_value, path_value) in zip(modules, old):
            if gate_value is missing:
                delattr(module, "history_gate")
            else:
                module.history_gate = gate_value
            if component_value is missing:
                delattr(module, "history_component_gates")
            else:
                module.history_component_gates = component_value
            if path_value is missing:
                delattr(module, "history_path_gates")
            else:
                module.history_path_gates = path_value


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
    return len(modules)


@contextmanager
def history_path_gates(pipeline, score, value):
    root = getattr(pipeline, "generator", pipeline)
    modules = [m for m in root.modules() if m.__class__.__name__ == "CausalWanSelfAttention"]
    missing = object()
    old = [(getattr(m, "history_gate", missing),
            getattr(m, "history_component_gates", missing),
            getattr(m, "history_path_gates", missing)) for m in modules]
    set_history_path_gates(pipeline, score, value)
    try:
        yield
    finally:
        for module, (gate_value, component_value, path_value) in zip(modules, old):
            for name, saved in (("history_gate", gate_value),
                                ("history_component_gates", component_value),
                                ("history_path_gates", path_value)):
                if saved is missing:
                    delattr(module, name)
                else:
                    setattr(module, name, saved)
