import torch

from wan.modules.attention import attention, gated_attention


def test_history_gate_changes_weights_without_scaling_keys():
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]]])
    v = torch.tensor([[[[10.0, 0.0]], [[0.0, 20.0]], [[30.0, 0.0]]]])
    full = gated_attention(q, k[:, :1], v[:, :1], k[:, 1:], v[:, 1:], 1.0, dtype=torch.float32)
    gated = gated_attention(q, k[:, :1], v[:, :1], k[:, 1:], v[:, 1:], 0.0, dtype=torch.float32)
    assert not torch.allclose(full, gated)
    assert torch.isfinite(gated).all()


def test_no_history_is_exact_for_all_gates():
    q = torch.randn(1, 2, 1, 4)
    k = torch.randn(1, 2, 1, 4)
    v = torch.randn(1, 2, 1, 4)
    baseline = attention(q, k, v, dtype=torch.float32)
    for gate in (0.75, 0.5, 0.25, 0.0):
        actual = gated_attention(q, k[:, :0], v[:, :0], k, v, gate, dtype=torch.float32)
        assert torch.equal(actual, baseline)


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
