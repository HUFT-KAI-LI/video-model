import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "code/LongLive")]
import torch

# Load the actual kernel file without importing unrelated Wan model dependencies.
import importlib.util
from unittest.mock import patch
_spec = importlib.util.spec_from_file_location("history_attention", ROOT / "code/LongLive/wan/modules/attention.py")
_kernel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_kernel)
attention, gated_attention = _kernel.attention, _kernel.gated_attention


class HistoryGateTests(unittest.TestCase):
    def setUp(self):
        # CPU regressions use the native SDPA path, even on hosts with FlashAttention installed.
        for name in ("FLASH_ATTN_2_AVAILABLE", "FLASH_ATTN_3_AVAILABLE"):
            patcher = patch.object(_kernel, name, False)
            patcher.start()
            self.addCleanup(patcher.stop)



    def test_history_gate_changes_weights_without_scaling_keys(self):
        q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        k = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 0.0]]]])
        v = torch.tensor([[[[10.0, 0.0]], [[0.0, 20.0]], [[30.0, 0.0]]]])
        full = gated_attention(q, k[:, :1], v[:, :1], k[:, 1:], v[:, 1:], 1.0, dtype=torch.float32)
        gated = gated_attention(q, k[:, :1], v[:, :1], k[:, 1:], v[:, 1:], 0.0, dtype=torch.float32)
        assert not torch.allclose(full, gated)
        assert torch.isfinite(gated).all()


    def test_no_history_is_exact_for_all_gates(self):
        q = torch.randn(1, 2, 1, 4)
        k = torch.randn(1, 2, 1, 4)
        v = torch.randn(1, 2, 1, 4)
        baseline = attention(q, k, v, dtype=torch.float32)
        for gate in (0.75, 0.5, 0.25, 0.0):
            actual = gated_attention(q, k[:, :0], v[:, :0], k, v, gate, dtype=torch.float32)
            assert torch.equal(actual, baseline)


    def test_gate_range_is_explicit(self):
        from restream.history_gate import set_history_gate
        class Empty:
            def modules(self): return []
        try:
            set_history_gate(Empty(), 1.2)
        except ValueError:
            pass
        else:
            raise AssertionError("out of range gate accepted")


    def test_full_history_gate_is_exact_native_attention(self):
        q = torch.randn(1, 2, 2, 8)
        k = torch.randn(1, 7, 2, 8)
        v = torch.randn(1, 7, 2, 8)
        baseline = attention(q, k, v, dtype=torch.float32)
        actual = gated_attention(q, k[:, :5], v[:, :5], k[:, 5:], v[:, 5:],
                                 1.0, dtype=torch.float32)
        assert torch.equal(actual, baseline)


    def test_context_restores_all_modules_nested_and_on_exception(self):
        from restream.history_gate import history_gate
        class CausalWanSelfAttention(torch.nn.Module):
            pass
        modules = torch.nn.ModuleList([CausalWanSelfAttention() for _ in range(3)])
        modules[0].history_gate = .75
        modules[1].history_gate = .25
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "injected"):
            with history_gate(modules, .5):
                assert all(m.history_gate == .5 for m in modules)
                with history_gate(modules, 0):
                    assert all(m.history_gate == 0 for m in modules)
                assert all(m.history_gate == .5 for m in modules)
                raise RuntimeError("injected")
        assert modules[0].history_gate == .75
        assert modules[1].history_gate == .25
        assert not hasattr(modules[2], "history_gate")


class CudaHistoryGateTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_native_kernel_exact_endpoints(self):
        q = torch.randn(1, 6, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 15, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        baseline = attention(q, k, v)
        self.assertTrue(torch.equal(baseline, gated_attention(q, k[:, :9], v[:, :9],
                                                              k[:, 9:], v[:, 9:], 1)))
        current = attention(q, k[:, 9:], v[:, 9:])
        for gate in (1, .75, .5, .25, 0):
            self.assertTrue(torch.equal(current, gated_attention(q, k[:, :0], v[:, :0],
                                                                 k[:, 9:], v[:, 9:], gate)))
