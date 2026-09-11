import torch

from wan.modules.attention import attention


def test_history_gate_changes_weights_without_scaling_keys():
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]]])
    v = torch.tensor([[[[10.0, 0.0]], [[0.0, 20.0]], [[30.0, 0.0]]]])
    full = attention(q, k, v, history_gate=1.0, history_tokens=1, dtype=torch.float32)
    gated = attention(q, k, v, history_gate=0.0, history_tokens=1, dtype=torch.float32)
    assert not torch.allclose(full, gated)
    assert torch.isfinite(gated).all()


def test_gate_range_is_explicit():
    from restream.history_gate import set_history_gate
    class Empty:
        def modules(self): return []
    try:
        set_history_gate(Empty(), 1.2)
    except ValueError:
        pass
    else:
        raise AssertionError("out of range gate accepted")
