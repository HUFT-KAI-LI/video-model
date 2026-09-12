"""D3 history attention-path protocol and analysis regressions."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream import edit_experiment as ex
from restream.history_path_analysis import analyze
from restream.history_paths import CONDITIONS, PROTOCOL, manifest_groups, validate_screen
from scripts import run_history_attention_path as runner


def config():
    return ex.read_config(ROOT / "configs/edit_ready_mvp.yaml")


def plan():
    return json.loads((ROOT / "configs/history_attention_path_plan.json").read_text())


def manifest():
    return json.loads((ROOT / "validation/history_attention_path_manifest.json").read_text())


def synthetic_records():
    values = {"full": .1, "global_.5": .4, "current_only": .9, "score_.5": .2,
              "value_.5": .3, "value_0": .5, "score_.5_value_.5": .7}
    records = []
    for entry in manifest()["cases"]:
        name = entry["condition"]
        spec = CONDITIONS[name]
        is_global = spec["kind"] == "global"
        records.append({
            "protocol": PROTOCOL, "prompt_id": entry["edit"], "seed": entry["seed"],
            "target_chunk": 4, "condition": name,
            "mechanism": "stage_c_global" if is_global else "attention_path",
            "history_gate": spec.get("gate"), "history_component_gates": None,
            "history_path_gates": None if is_global else {
                "score": spec["score_gate"], "value": spec["value_gate"]},
            "evidence": "directional", "D_drift": 0.0 if name == "full" else .1,
            "cache": {"sha256": f"checkpoint-{entry['edit']}"},
            "chunk_latent_sha256": {"replay": "base" if name == "full" else f"{name}-p0",
                                    "text_rebind": f"{name}-p1"},
            "history_partition": None if is_global else {
                p: {"latent_frames": {"history": 9, "current": 3}}
                for p in ("replay", "text_rebind")},
            "rng": {"replay": {"exact": True}, "text_rebind": {"exact": True}},
            "responsiveness": {"replay": {"S_proxy": 0.0},
                               "text_rebind": {"S_proxy": values[name]},
                               "full_regeneration": {"S_proxy": 1.0}},
            "editability": {"E": values[name], "S_full": 1.0, "R": values[name]},
            "sanity": {"fixed_references_unmodified_full_history": True,
                       "full_history_P0_exact_base": True if name == "full" else None},
            "preservation": {"outside_exact": True,
                             "identity": {"status": "measured",
                                          "replay": {"mean_cosine": .9,
                                                     "mean_cosine_distance": .1},
                                          "text_rebind": {"mean_cosine": .8,
                                                          "mean_cosine_distance": .2}}},
            "boundary": {"replay": {}, "text_rebind": {}},
            "cost": {"status": "invalid_diagnostic"}})
    return records


class HistoryPathTests(unittest.TestCase):
    def test_frozen_manifest_is_exact_28_pair_grid(self):
        groups = manifest_groups(config(), manifest())
        validate_screen(groups, plan())
        self.assertEqual(len(groups), 4)
        self.assertEqual({group["seed"] for group in groups}, {505})
        self.assertEqual(sum(len(v) for g in groups
                             for v in g["conditions_by_target"].values()), 28)

    def test_analyzer_reports_frozen_path_contrasts(self):
        report = analyze(synthetic_records(), plan(), [{"source": "synthetic"}])
        unit = report["path_units"][0]
        self.assertAlmostEqual(unit["score_value_inclusion_exclusion_contrast"], .3)
        self.assertAlmostEqual(unit["value_zero_competition_gap"], .4)
        self.assertEqual(report["decision"],
                         "descriptive_only_no_automatic_path_classification")

    def test_analyzer_fails_closed_on_path_partition_error(self):
        records = synthetic_records()
        record = next(row for row in records if row["condition"] == "value_.5")
        record["history_partition"]["replay"]["latent_frames"]["history"] = 6
        with self.assertRaisesRegex(ValueError, "path partition audit"):
            analyze(records, plan(), [])

    def test_analyzer_fails_closed_on_inert_path(self):
        records = synthetic_records()
        edit = "dress_red_to_blue"
        full = next(row for row in records if row["prompt_id"] == edit and row["condition"] == "full")
        score = next(row for row in records if row["prompt_id"] == edit and row["condition"] == "score_.5")
        score["chunk_latent_sha256"] = copy.deepcopy(full["chunk_latent_sha256"])
        with self.assertRaisesRegex(ValueError, "score_.5 is inert"):
            analyze(records, plan(), [])

    def test_runner_binds_full_path_protocol_and_restores_context(self):
        class CausalWanSelfAttention(torch.nn.Module):
            pass
        module = CausalWanSelfAttention()
        module.history_gate = .25
        pipeline = SimpleNamespace(generator=torch.nn.ModuleList([module]))

        def fixed(*args, **kwargs):
            self.assertEqual(module.history_gate, 1.0)
            self.assertEqual(kwargs["protocol"], PROTOCOL)
            self.assertEqual(kwargs["full_condition"], "full")
            self.assertEqual(kwargs["spec_fn"]("score_.5"),
                             {"kind": "path", "score_gate": .5, "value_gate": 1.0})
            return ["record"]

        with patch.object(runner, "_run_fixed_group", fixed):
            result = runner.run_group(pipeline, {}, {}, "cpu", None, None, {}, None)
        self.assertEqual(result, ["record"])
        self.assertEqual(module.history_gate, .25)


if __name__ == "__main__":
    unittest.main()
