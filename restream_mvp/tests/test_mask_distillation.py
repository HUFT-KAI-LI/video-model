"""M1-A feature, controller, manifest, and decision tests."""
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream import edit_experiment as ex
from restream.mask_distillation import (PromptOnlyController, PromptStateController,
                                        checkpoint_state_feature)
from restream.mask_distillation_analysis import analyze
from restream.mask_distillation_protocol import (EDITS, FINAL_CONDITIONS, HELDOUT_SEEDS,
                                                  groups_from_manifest)


class MaskDistillationTests(unittest.TestCase):
    def test_manifests_are_disjoint_and_complete(self):
        config = ex.read_config(ROOT / "configs/edit_ready_mvp.yaml")
        teacher = json.loads((ROOT / "validation/mask_distillation_teacher_manifest.json").read_text())
        heldout = json.loads((ROOT / "validation/mask_distillation_heldout_manifest.json").read_text())
        train_groups = groups_from_manifest(config, teacher, "mask_distillation_teacher_m1a", tuple(range(801, 809)))
        test_groups = groups_from_manifest(config, heldout, "mask_distillation_heldout_m1a", HELDOUT_SEEDS)
        self.assertEqual((len(train_groups), len(test_groups)), (32, 8))
        self.assertFalse({g["seed"] for g in train_groups} & {g["seed"] for g in test_groups})

    def test_state_pool_uses_sink_and_surviving_local_history(self):
        frame_tokens = 2
        layers = []
        for _ in range(30):
            value = torch.zeros(1, 24, 12, 128)
            value[:, :6] = 2
            value[:, 6:12] = 100  # evicted local frames must not enter the summary
            value[:, 12:24] = 4
            layers.append({"v": value, "local_end_index": torch.tensor([24])})
        checkpoint = {"kv_cache": layers, "frame_seq_length": frame_tokens,
                      "sink_size": 3, "local_attn_size": 12, "num_frame_per_block": 3}
        feature = checkpoint_state_feature(checkpoint)
        self.assertEqual(tuple(feature.shape), (30, 24))
        expected_mean = (3 * 2 + 6 * 4) / 9
        self.assertTrue(torch.allclose(feature[:, :12], torch.full((30, 12), expected_mean)))
        self.assertLess(float(feature.max()), 5)

    def test_controller_shapes_and_bounds(self):
        prompt = torch.randn(5, 64); state = torch.randn(5, 30, 24)
        for output in (PromptOnlyController()(prompt), PromptStateController()(prompt, state)):
            self.assertEqual(tuple(output.shape), (5, 30))
            self.assertTrue(bool(((output > 0) & (output < 1)).all()))

    def test_analyzer_requires_six_heldout_pareto_wins(self):
        plan = json.loads((ROOT / "configs/mask_distillation_plan.json").read_text())
        records = []
        for edit in EDITS:
            for seed in HELDOUT_SEEDS:
                unit_index = list(EDITS).index(edit) * 2 + list(HELDOUT_SEEDS).index(seed)
                for condition in FINAL_CONDITIONS:
                    delta, drift = {"full": (0., 0.), "global_.5": (.2, .2),
                                    "current_only": (.8, .8), "prompt_only": (.1, .1),
                                    "prompt_state": ((.3, .15) if unit_index < 6 else (.1, .3)),
                                    "oracle": (.4, .1)}[condition]
                    mask = ([0.] * 30 if condition == "full" else [1.] * 30
                            if condition == "current_only" else [.5] * 30)
                    records.append({"protocol": plan["protocol"], "prompt_id": edit, "seed": seed,
                                    "target_chunk": 4, "condition": condition, "layer_release": mask,
                                    "checkpoint_sha256": f"{edit}-{seed}", "D_drift": drift,
                                    "editability": {"E": delta, "R": delta}, "drift": {"exact": condition == "full"},
                                    "responsiveness": {"replay": {"S_proxy": 0.}, "text_rebind": {"S_proxy": delta}},
                                    "rng": {name: {"exact": True} for name in ("replay", "text_rebind")},
                                    "history_partition": {name: {"latent_frames": {"history": 9, "current": 3},
                                                                          "modules_checked": 30}
                                                          for name in ("replay", "text_rebind")},
                                    "preservation": {"outside_exact": True, "identity": {"status": "measured"}}})
        report = analyze(records, plan, [])
        self.assertEqual(report["decision"], "PASS")
        self.assertEqual(report["pareto_wins"]["prompt_state"], 6)
        self.assertEqual(report["passing_controllers"], ["prompt_state"])


if __name__ == "__main__": unittest.main()
