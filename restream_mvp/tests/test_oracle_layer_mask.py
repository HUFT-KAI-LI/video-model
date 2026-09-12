"""M0/M1 oracle layer-release protocol regressions."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream import edit_experiment as ex
from restream.history_gate import history_layer_release
from restream.oracle_layer_analysis import analyze
from restream.oracle_layer_mask import PROTOCOL, manifest_groups, validate_plan


def _plan():
    return json.loads((ROOT / "configs/oracle_layer_mask_plan.json").read_text())


def _manifest():
    return json.loads((ROOT / "validation/oracle_layer_mask_manifest.json").read_text())


class OracleLayerMaskTests(unittest.TestCase):
    def test_manifest_is_frozen_eight_unit_grid(self):
        groups = manifest_groups(ex.read_config(ROOT / "configs/edit_ready_mvp.yaml"), _manifest())
        validate_plan(groups, _plan())
        self.assertEqual(len(groups), 8)
        self.assertEqual({group["seed"] for group in groups}, {606, 707})

    def test_attention_release_endpoints_and_half_match_stage_c(self):
        spec = importlib.util.spec_from_file_location(
            "oracle_attention", ROOT / "code/LongLive/wan/modules/attention.py")
        kernel = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel)
        original = kernel.attention
        kernel.attention = lambda q, k, v, **kwargs: v.mean(dim=1, keepdim=True).expand_as(q)
        try:
            q = torch.zeros(1, 2, 1, 1)
            kh, vh = torch.zeros(1, 2, 1, 1), torch.tensor([[[[2.]], [[4.]]]])
            kc, vc = torch.zeros(1, 2, 1, 1), torch.tensor([[[[6.]], [[8.]]]])
            full = kernel.attention(q, torch.cat([kh, kc], 1), torch.cat([vh, vc], 1))
            current = kernel.attention(q, kc, vc)
            self.assertTrue(torch.equal(kernel.layer_release_attention(q, kh, vh, kc, vc, 0), full))
            self.assertTrue(torch.equal(kernel.layer_release_attention(q, kh, vh, kc, vc, 1), current))
            self.assertTrue(torch.equal(kernel.layer_release_attention(q, kh, vh, kc, vc, .5),
                                        kernel.gated_attention(q, kh, vh, kc, vc, .5)))
        finally:
            kernel.attention = original

    def test_layer_context_assigns_in_order_and_restores_every_state(self):
        class CausalWanSelfAttention(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.history_gate = value
                self.history_component_gates = {"old": value}
                self.history_path_gates = {"score": value}
                self.history_layer_release = None
        modules = [CausalWanSelfAttention(index / 30) for index in range(30)]
        pipeline = SimpleNamespace(generator=torch.nn.ModuleList(modules))
        before = [(m.history_gate, m.history_component_gates, m.history_path_gates,
                   m.history_layer_release) for m in modules]
        mask = [index / 29 for index in range(30)]
        with history_layer_release(pipeline, mask):
            self.assertEqual([m.history_layer_release for m in modules], mask)
            self.assertTrue(all(m.history_component_gates is None for m in modules))
        after = [(m.history_gate, m.history_component_gates, m.history_path_gates,
                  m.history_layer_release) for m in modules]
        self.assertEqual(after, before)

    def test_analyzer_applies_pareto_rule(self):
        plan = _plan()
        records = []
        conditions = plan["manifest"]["final_conditions"]
        for edit in plan["manifest"]["edits"]:
            for seed in plan["manifest"]["seeds"]:
                for condition in conditions:
                    delta, drift = (0.4, 0.1) if condition == "global_.5" else (0.0, 0.0)
                    if condition == "oracle_lambda_0.2" and edit == plan["manifest"]["edits"][0] and seed == 606:
                        delta, drift = 0.5, 0.09
                    mask = ([0.] * 30 if condition == "full" else [1.] * 30
                            if condition == "current_only" else [0.5] * 30)
                    exact = condition == "full"
                    records.append({"protocol": PROTOCOL, "prompt_id": edit, "seed": seed,
                                    "target_chunk": 4, "condition": condition,
                                    "layer_release": mask, "checkpoint_sha256": f"{edit}-{seed}",
                                    "chunk_latent_sha256": {
                                        name: ("full" if condition == "full" else condition)
                                        for name in ("replay", "text_rebind")},
                                    "editability": {"E": delta, "R": delta}, "D_drift": drift,
                                    "drift": {"exact": exact},
                                    "responsiveness": {"replay": {"S_proxy": 0.0},
                                                       "text_rebind": {"S_proxy": delta}},
                                    "rng": {name: {"exact": True} for name in ("replay", "text_rebind")},
                                    "history_partition": {name: {"latent_frames": {"history": 9, "current": 3},
                                                                          "modules_checked": 30}
                                                          for name in ("replay", "text_rebind")},
                                    "preservation": {"outside_exact": True,
                                                     "identity": {"status": "measured"}}})
        report = analyze(records, plan, [])
        self.assertEqual(report["decision"], "PASS")
        self.assertEqual(report["units_with_oracle_pareto_gain"], 1)


if __name__ == "__main__":
    unittest.main()
