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
component_gated_attention = _kernel.component_gated_attention
path_gated_attention = _kernel.path_gated_attention


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

    def test_component_full_history_is_exact_native_attention(self):
        q = torch.randn(1, 2, 2, 8)
        k = torch.randn(1, 9, 2, 8)
        v = torch.randn(1, 9, 2, 8)
        components = {"sink": (k[:, :2], v[:, :2]),
                      "old": (k[:, 2:4], v[:, 2:4]),
                      "recent": (k[:, 4:7], v[:, 4:7])}
        actual = component_gated_attention(
            q, components, k[:, 7:], v[:, 7:],
            {"sink": 1, "old": 1, "recent": 1}, dtype=torch.float32)
        assert torch.equal(actual, attention(q, k, v, dtype=torch.float32))

    def test_all_binary_component_subsets_are_exact_native_attention(self):
        q = torch.randn(1, 2, 2, 8)
        k = torch.randn(1, 9, 2, 8)
        v = torch.randn(1, 9, 2, 8)
        components = {"sink": (k[:, :2], v[:, :2]),
                      "old": (k[:, 2:4], v[:, 2:4]),
                      "recent": (k[:, 4:6], v[:, 4:6])}
        current_k, current_v = k[:, 6:], v[:, 6:]
        for sink in (0, 1):
            for old in (0, 1):
                for recent in (0, 1):
                    gates = {"sink": sink, "old": old, "recent": recent}
                    included = [components[name] for name in ("sink", "old", "recent")
                                if gates[name]]
                    expected_k = torch.cat([part[0] for part in included] + [current_k], dim=1)
                    expected_v = torch.cat([part[1] for part in included] + [current_v], dim=1)
                    actual = component_gated_attention(
                        q, components, current_k, current_v, gates, dtype=torch.float32)
                    self.assertTrue(torch.equal(
                        actual, attention(q, expected_k, expected_v, dtype=torch.float32)))

    def test_single_component_gate_interpolates_native_outputs(self):
        q = torch.randn(1, 2, 1, 4)
        k = torch.randn(1, 8, 1, 4)
        v = torch.randn(1, 8, 1, 4)
        components = {"sink": (k[:, :2], v[:, :2]),
                      "old": (k[:, 2:4], v[:, 2:4]),
                      "recent": (k[:, 4:6], v[:, 4:6])}
        full = attention(q, k, v, dtype=torch.float32)
        without_sink = attention(q, k[:, 2:], v[:, 2:], dtype=torch.float32)
        actual = component_gated_attention(
            q, components, k[:, 6:], v[:, 6:],
            {"sink": .5, "old": 1, "recent": 1}, dtype=torch.float32)
        assert torch.equal(actual, without_sink + .5 * (full - without_sink))

    def test_path_score_gate_matches_explicit_log_odds_bias(self):
        q = torch.randn(1, 2, 2, 4)
        kh = torch.randn(1, 3, 2, 4)
        vh = torch.randn(1, 3, 2, 4)
        kc = torch.randn(1, 2, 2, 4)
        vc = torch.randn(1, 2, 2, 4)
        actual = path_gated_attention(
            q, kh, vh, kc, vc, .5, .25, dtype=torch.float32)
        qt = q.transpose(1, 2)
        kt = torch.cat([kh, kc], 1).transpose(1, 2)
        vt = torch.cat([vh * .25, vc], 1).transpose(1, 2)
        logits = torch.matmul(qt, kt.transpose(-2, -1)) / (q.shape[-1] ** .5)
        logits[..., :kh.shape[1]] += torch.log(torch.tensor(.5))
        expected = (torch.softmax(logits, -1) @ vt).transpose(1, 2).contiguous()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_path_exact_endpoints_and_value_zero_keeps_competition(self):
        q = torch.randn(1, 2, 1, 4)
        kh = torch.randn(1, 3, 1, 4)
        vh = torch.randn(1, 3, 1, 4)
        kc = torch.randn(1, 2, 1, 4)
        vc = torch.randn(1, 2, 1, 4)
        full = attention(q, torch.cat([kh, kc], 1), torch.cat([vh, vc], 1),
                         dtype=torch.float32)
        current = attention(q, kc, vc, dtype=torch.float32)
        self.assertTrue(torch.equal(full, path_gated_attention(
            q, kh, vh, kc, vc, 1, 1, dtype=torch.float32)))
        self.assertTrue(torch.equal(current, path_gated_attention(
            q, kh, vh, kc, vc, 0, 1, dtype=torch.float32)))
        value_zero = path_gated_attention(q, kh, vh, kc, vc, 1, 0, dtype=torch.float32)
        self.assertFalse(torch.equal(value_zero, current))

    def test_sink_only_gate_matches_global_gate_when_other_history_is_empty(self):
        q = torch.randn(1, 3, 2, 8)
        k = torch.randn(1, 6, 2, 8)
        v = torch.randn(1, 6, 2, 8)
        components = {"sink": (k[:, :3], v[:, :3]),
                      "old": (k[:, :0], v[:, :0]),
                      "recent": (k[:, :0], v[:, :0])}
        global_output = gated_attention(
            q, k[:, :3], v[:, :3], k[:, 3:], v[:, 3:], .5, dtype=torch.float32)
        sink_output = component_gated_attention(
            q, components, k[:, 3:], v[:, 3:],
            {"sink": .5, "old": 1, "recent": 1}, dtype=torch.float32)
        assert torch.equal(global_output, sink_output)

    def test_empty_component_gate_is_exact_noop(self):
        q = torch.randn(1, 2, 1, 4)
        k = torch.randn(1, 5, 1, 4)
        v = torch.randn(1, 5, 1, 4)
        components = {"sink": (k[:, :3], v[:, :3]),
                      "old": (k[:, :0], v[:, :0]),
                      "recent": (k[:, :0], v[:, :0])}
        expected = attention(q, torch.cat([k[:, :3], k[:, 3:]], 1),
                             torch.cat([v[:, :3], v[:, 3:]], 1), dtype=torch.float32)
        for gates in ({"sink": 1, "old": .5, "recent": 1},
                      {"sink": 1, "old": 1, "recent": .5},
                      {"sink": 1, "old": .5, "recent": .5}):
            actual = component_gated_attention(
                q, components, k[:, 3:], v[:, 3:], gates, dtype=torch.float32)
            assert torch.equal(actual, expected)


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

    def test_component_context_restores_global_and_component_state(self):
        from restream.history_gate import history_component_gates
        class CausalWanSelfAttention(torch.nn.Module):
            pass
        modules = torch.nn.ModuleList([CausalWanSelfAttention() for _ in range(2)])
        modules[0].history_gate = .25
        modules[0].history_component_gates = {"sink": .2, "old": .3, "recent": .4}
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "injected"):
            with history_component_gates(
                    modules, {"sink": .5, "old": 1, "recent": 1}):
                assert all(module.history_gate == 1 for module in modules)
                assert all(module.history_component_gates["sink"] == .5 for module in modules)
                raise RuntimeError("injected")
        assert modules[0].history_gate == .25
        assert modules[0].history_component_gates == {"sink": .2, "old": .3, "recent": .4}
        assert not hasattr(modules[1], "history_gate")
        assert not hasattr(modules[1], "history_component_gates")

    def test_path_context_restores_all_gate_state(self):
        from restream.history_gate import history_path_gates
        class CausalWanSelfAttention(torch.nn.Module):
            pass
        module = CausalWanSelfAttention()
        module.history_gate = .25
        module.history_component_gates = {"sink": .2, "old": .3, "recent": .4}
        module.history_path_gates = {"score": .75, "value": .5}
        with history_path_gates(torch.nn.ModuleList([module]), .5, 0):
            self.assertEqual(module.history_gate, 1)
            self.assertIsNone(module.history_component_gates)
            self.assertEqual(module.history_path_gates, {"score": .5, "value": 0.0})
        self.assertEqual(module.history_gate, .25)
        self.assertEqual(module.history_component_gates,
                         {"sink": .2, "old": .3, "recent": .4})
        self.assertEqual(module.history_path_gates, {"score": .75, "value": .5})


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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_component_kernel_native_paths(self):
        q = torch.randn(1, 6, 2, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 18, 2, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        components = {"sink": (k[:, :3], v[:, :3]),
                      "old": (k[:, 3:6], v[:, 3:6]),
                      "recent": (k[:, 6:12], v[:, 6:12])}
        full = attention(q, k, v)
        actual = component_gated_attention(
            q, components, k[:, 12:], v[:, 12:],
            {"sink": 1, "old": 1, "recent": 1})
        self.assertTrue(torch.equal(actual, full))
        selective = component_gated_attention(
            q, components, k[:, 12:], v[:, 12:],
            {"sink": .5, "old": 1, "recent": 1})
        self.assertTrue(torch.isfinite(selective).all())
        self.assertFalse(torch.equal(selective, full))
        sink_only = {"sink": (k[:, :6], v[:, :6]),
                     "old": (k[:, :0], v[:, :0]),
                     "recent": (k[:, :0], v[:, :0])}
        global_output = gated_attention(q, k[:, :6], v[:, :6],
                                        k[:, 12:], v[:, 12:], .5)
        sink_output = component_gated_attention(
            q, sink_only, k[:, 12:], v[:, 12:],
            {"sink": .5, "old": 1, "recent": 1})
        self.assertTrue(torch.equal(global_output, sink_output))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_path_kernel_native_endpoints_and_biased_path(self):
        q = torch.randn(1, 6, 2, 64, device="cuda", dtype=torch.bfloat16)
        kh = torch.randn(1, 9, 2, 64, device="cuda", dtype=torch.bfloat16)
        vh = torch.randn_like(kh)
        kc = torch.randn(1, 6, 2, 64, device="cuda", dtype=torch.bfloat16)
        vc = torch.randn_like(kc)
        current = attention(q, kc, vc)
        full = attention(q, torch.cat([kh, kc], 1), torch.cat([vh, vc], 1))
        self.assertTrue(torch.equal(current, path_gated_attention(q, kh, vh, kc, vc, 0, 1)))
        self.assertTrue(torch.equal(full, path_gated_attention(q, kh, vh, kc, vc, 1, 1)))
        biased = path_gated_attention(q, kh, vh, kc, vc, .5, .5)
        self.assertTrue(torch.isfinite(biased).all())
        self.assertFalse(torch.equal(biased, full))
