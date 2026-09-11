"""D2 history-subset protocol and factorial-analysis regressions."""
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
from restream.history_subset_analysis import analyze
from restream.history_subsets import CONDITIONS, PROTOCOL, manifest_groups, validate_screen
from scripts import run_history_subset_interaction as runner


def config():
    return ex.read_config(ROOT / "configs/edit_ready_mvp.yaml")


def plan():
    return json.loads((ROOT / "configs/history_subset_interaction_plan.json").read_text())


def manifest():
    return json.loads((ROOT / "validation/history_subset_interaction_manifest.json").read_text())


def synthetic_records():
    values = {"empty": 0.0, "S": 1.0, "O": 2.0, "R": 4.0,
              "SO": 5.0, "SR": 8.0, "OR": 10.0, "SOR": 20.0,
              "global_release": 12.0}
    records = []
    for entry in manifest()["cases"]:
        name = entry["condition"]
        spec = CONDITIONS[name]
        records.append({
            "protocol": PROTOCOL, "prompt_id": entry["edit"], "seed": entry["seed"],
            "target_chunk": 4, "condition": name,
            "mechanism": "stage_c_global" if spec["kind"] == "global" else "component",
            "history_gate": spec.get("gate"),
            "history_component_gates": copy.deepcopy(spec.get("gates")),
            "evidence": "directional", "D_drift": 0.0 if name == "SOR" else 0.1,
            "cache": {"sha256": f"checkpoint-{entry['edit']}"},
            "chunk_latent_sha256": {
                "replay": "base" if name == "SOR" else f"{name}-p0",
                "text_rebind": f"{name}-p1"},
            "history_partition": None if name == "global_release" else {
                policy: {"latent_frames": {"sink": 3, "old": 3, "recent": 3, "current": 3}}
                for policy in ("replay", "text_rebind")},
            "rng": {"replay": {"exact": True}, "text_rebind": {"exact": True}},
            "responsiveness": {"replay": {"S_proxy": 0.0},
                               "text_rebind": {"S_proxy": values[name]},
                               "full_regeneration": {"S_proxy": 1.0}},
            "editability": {"E": values[name], "S_full": 1.0, "R": values[name]},
            "sanity": {"fixed_references_unmodified_full_history": True,
                       "full_history_P0_exact_base": True if name == "SOR" else None},
            "preservation": {"outside_exact": True,
                             "identity": {"status": "measured",
                                          "replay": {"mean_cosine": .9,
                                                     "mean_cosine_distance": .1},
                                          "text_rebind": {"mean_cosine": .8,
                                                          "mean_cosine_distance": .2}}},
            "boundary": {"replay": {}, "text_rebind": {}},
            "cost": {"status": "invalid_diagnostic"}})
    return records


class HistorySubsetTests(unittest.TestCase):
    def test_frozen_manifest_is_exact_36_pair_grid(self):
        groups = manifest_groups(config(), manifest())
        validate_screen(groups, plan())
        self.assertEqual(len(groups), 4)
        self.assertEqual({group["seed"] for group in groups}, {404})
        self.assertEqual(sum(len(values) for group in groups
                             for values in group["conditions_by_target"].values()), 36)

    def test_manifest_rejects_nonbinary_component_gate(self):
        original = CONDITIONS["S"]["gates"]["sink"]
        CONDITIONS["S"]["gates"]["sink"] = .5
        try:
            with self.assertRaisesRegex(ValueError, "binary"):
                manifest_groups(config(), manifest())
        finally:
            CONDITIONS["S"]["gates"]["sink"] = original

    def test_factorial_interactions_and_shapley_are_exact(self):
        report = analyze(synthetic_records(), plan(), [{"source": "synthetic"}])
        unit = report["factorial_units"][0]
        self.assertEqual(unit["pairwise_and_third_order_interactions"],
                         {"I_SO": 2.0, "I_SR": 3.0, "I_OR": 4.0, "I_SOR": 4.0})
        self.assertAlmostEqual(sum(unit["locking_shapley"].values()),
                               unit["locking"]["SOR"])
        self.assertEqual(report["decision"],
                         "descriptive_only_no_automatic_pattern_classification")
        self.assertNotIn("global_release", unit["F_R"])

    def test_analyzer_fails_closed_on_partition_error(self):
        records = synthetic_records()
        record = next(row for row in records if row["condition"] == "SO")
        record["history_partition"]["replay"]["latent_frames"]["old"] = 0
        with self.assertRaisesRegex(ValueError, "partition audit"):
            analyze(records, plan(), [])

    def test_analyzer_fails_closed_on_inert_current_only(self):
        records = synthetic_records()
        full = next(row for row in records
                    if row["prompt_id"] == "dress_red_to_blue" and row["condition"] == "SOR")
        empty = next(row for row in records
                     if row["prompt_id"] == "dress_red_to_blue" and row["condition"] == "empty")
        empty["chunk_latent_sha256"] = copy.deepcopy(full["chunk_latent_sha256"])
        with self.assertRaisesRegex(ValueError, "current-only endpoint is inert"):
            analyze(records, plan(), [])

    def test_runner_binds_d2_protocol_inside_full_history_context(self):
        class CausalWanSelfAttention(torch.nn.Module):
            pass
        module = CausalWanSelfAttention()
        module.history_gate = .25
        pipeline = SimpleNamespace(generator=torch.nn.ModuleList([module]))

        def fixed(*args, **kwargs):
            self.assertEqual(module.history_gate, 1.0)
            self.assertEqual(kwargs["protocol"], PROTOCOL)
            self.assertEqual(kwargs["full_condition"], "SOR")
            self.assertEqual(kwargs["spec_fn"]("empty")["gates"],
                             {"sink": 0.0, "old": 0.0, "recent": 0.0})
            return ["record"]

        with patch.object(runner, "_run_fixed_group", fixed):
            result = runner.run_group(pipeline, {}, {}, "cpu", None, None, {}, None)
        self.assertEqual(result, ["record"])
        self.assertEqual(module.history_gate, .25)


if __name__ == "__main__":
    unittest.main()
