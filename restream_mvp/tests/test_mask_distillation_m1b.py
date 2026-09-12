"""Frozen M1-B grid and decision tests."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream import edit_experiment as ex
from restream.mask_distillation import PromptOnlyController, PromptStateController
from restream.mask_distillation_analysis import analyze
from restream.mask_distillation_protocol import (EDITS, M1B_FINAL_CONDITIONS,
                                                  M1B_HELDOUT_SEEDS,
                                                  M1B_NEW_TEACHER_SEEDS,
                                                  groups_from_manifest)


class MaskDistillationM1BTests(unittest.TestCase):
    def test_controller_architecture_is_unchanged(self):
        self.assertEqual(sum(p.numel() for p in PromptOnlyController().parameters()), 6110)
        self.assertEqual(sum(p.numel() for p in PromptStateController().parameters()), 6513)

    def test_frozen_grids_are_complete_and_disjoint(self):
        config = ex.read_config(ROOT / "configs/edit_ready_mvp.yaml")
        teacher = json.loads((ROOT / "validation/mask_distillation_m1b_teacher_manifest.json").read_text())
        heldout = json.loads((ROOT / "validation/mask_distillation_m1b_heldout_manifest.json").read_text())
        train = groups_from_manifest(config, teacher, "mask_distillation_teacher_m1b",
                                     M1B_NEW_TEACHER_SEEDS)
        test = groups_from_manifest(config, heldout, "mask_distillation_heldout_m1b",
                                    M1B_HELDOUT_SEEDS)
        self.assertEqual((len(train), len(test)), (160, 16))
        self.assertFalse({row["seed"] for row in train} & {row["seed"] for row in test})

    def test_only_prompt_state_can_trigger_twelve_of_sixteen_pass(self):
        plan = json.loads((ROOT / "configs/mask_distillation_m1b_plan.json").read_text())
        records = []
        for edit_index, edit in enumerate(EDITS):
            for seed_index, seed in enumerate(M1B_HELDOUT_SEEDS):
                unit_index = edit_index * 4 + seed_index
                for condition in M1B_FINAL_CONDITIONS:
                    delta, drift = {"full": (0., 0.), "global_.5": (.2, .2),
                                    "prompt_only": (.3, .1),
                                    "prompt_state": ((.3, .15) if unit_index < 12 else (.1, .3)),
                                    "oracle": (.4, .1)}[condition]
                    mask = [0.] * 30 if condition == "full" else [.5] * 30
                    records.append({"protocol": plan["protocol"], "prompt_id": edit, "seed": seed,
                                    "target_chunk": 4, "condition": condition, "layer_release": mask,
                                    "checkpoint_sha256": f"{edit}-{seed}", "D_drift": drift,
                                    "editability": {"E": delta, "R": delta},
                                    "drift": {"exact": condition == "full"},
                                    "responsiveness": {"replay": {"S_proxy": 0.},
                                                       "text_rebind": {"S_proxy": delta}},
                                    "rng": {name: {"exact": True} for name in ("replay", "text_rebind")},
                                    "history_partition": {name: {"latent_frames": {"history": 9, "current": 3},
                                                                          "modules_checked": 30}
                                                          for name in ("replay", "text_rebind")},
                                    "preservation": {"outside_exact": True,
                                                     "identity": {"status": "measured"}}})
        report = analyze(records, plan, [])
        self.assertEqual(report["decision"], "PASS")
        self.assertEqual(report["pareto_wins"], {"prompt_only": 16, "prompt_state": 12})
        self.assertEqual(report["primary_controller"], "prompt_state")


if __name__ == "__main__":
    unittest.main()
