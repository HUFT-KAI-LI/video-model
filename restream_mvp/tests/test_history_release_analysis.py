"""Regression tests for the preregistered paired/Pareto decision."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream.history_release_analysis import analyze, exact_positive_sign_p
from restream import edit_cache as ec


def load_plan():
    return json.loads((ROOT / "configs/history_release_analysis_plan.json").read_text())


def synthetic_records(negative_clusters=1):
    plan = load_plan()
    clusters = [(edit, seed) for edit in plan["manifest"]["edits"]
                for seed in plan["manifest"]["seeds"]]
    records = []
    for edit, seed in clusters:
        negative = clusters.index((edit, seed)) >= len(clusters) - negative_clusters
        for chunk in [0, 1, 4]:
            for gate in plan["manifest"]["gates"]:
                if chunk == 0 or gate == 1:
                    delta = 0.0
                elif gate == .5:
                    delta = -.02 if negative else .02
                else:
                    delta = {0.75: .01, 0.25: .04, 0.0: .06}[gate]
                e = .1 + delta
                drift = 0.0 if chunk == 0 or gate == 1 else (1 - gate) * .2
                suffix = "calibration" if chunk == 0 else str(gate)
                records.append({
                    "protocol": plan["protocol"], "prompt_id": edit, "seed": seed,
                    "target_chunk": chunk, "history_gate": gate,
                    "reference_history_gate": 1.0, "evidence": "directional",
                    "responsiveness": {"replay": {"S_proxy": .03},
                                       "text_rebind": {"S_proxy": .03 + e}},
                    "editability": {"E": e}, "D_drift": drift,
                    "preservation": {"outside_exact": True},
                    "rng": {"replay": {"exact": True}, "text_rebind": {"exact": True}},
                    "sanity": {"g1_P0_exact_base": gate == 1 if gate == 1 else None,
                               "g1_P1_exact_sealed": True if gate == 1 and edit ==
                               plan["manifest"]["edits"][0] and seed == 42 else None},
                    "chunk_latent_sha256": {"replay": f"p0-{edit}-{seed}-{suffix}",
                                             "text_rebind": f"p1-{edit}-{seed}-{suffix}"}})
    return records


class HistoryReleaseAnalysisTests(unittest.TestCase):
    def test_frozen_primary_threshold_is_exact(self):
        plan = load_plan()
        self.assertEqual(ec.sha256_file(ROOT / plan["manifest"]["path"]),
                         plan["manifest"]["sha256"])
        self.assertEqual(plan["confirmatory"]["gate"], .5)
        self.assertEqual(plan["confirmatory"]["minimum_positive_clusters"], 7)
        self.assertIsNone(plan["confirmatory"]["magnitude_threshold"])
        self.assertAlmostEqual(exact_positive_sign_p(7, 1), .03515625)
        self.assertGreater(exact_positive_sign_p(6, 2), .05)

    def test_seven_of_eight_with_replication_passes(self):
        report = analyze(synthetic_records(negative_clusters=1), load_plan())
        self.assertTrue(report["confirmatory"]["passed"])
        self.assertEqual(report["confirmatory"]["positive"], 7)
        self.assertAlmostEqual(report["confirmatory"]["exact_one_sided_sign_p"], .03515625)
        self.assertEqual(set(report["pareto"]["frontier_gates"]), {1, .75, .5, .25, 0})
        self.assertIsNone(report["pareto"]["operating_point"])

    def test_six_of_eight_fails_confirmatory_rule(self):
        report = analyze(synthetic_records(negative_clusters=2), load_plan())
        self.assertFalse(report["confirmatory"]["passed"])
        self.assertFalse(report["confirmatory"]["criteria"]["positive_clusters"])
        self.assertFalse(report["confirmatory"]["criteria"]["sign_test"])

    def test_missing_duplicate_and_broken_invariant_fail_closed(self):
        records, plan = synthetic_records(), load_plan()
        with self.assertRaisesRegex(ValueError, "incomplete case grid"):
            analyze(records[:-1], plan)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            analyze(records + [records[0]], plan)
        records[0]["preservation"]["outside_exact"] = False
        with self.assertRaisesRegex(ValueError, "outside preservation"):
            analyze(records, plan)

    def test_dominated_gate_is_removed_without_selecting_winner(self):
        records, plan = synthetic_records(), load_plan()
        for record in records:
            if record["target_chunk"] > 0 and record["history_gate"] == .25:
                reference = next(row for row in records
                                 if row["prompt_id"] == record["prompt_id"] and
                                 row["seed"] == record["seed"] and
                                 row["target_chunk"] == record["target_chunk"] and
                                 row["history_gate"] == .5)
                record["editability"]["E"] = reference["editability"]["E"]
                record["responsiveness"]["text_rebind"]["S_proxy"] = \
                    record["responsiveness"]["replay"]["S_proxy"] + record["editability"]["E"]
        report = analyze(records, plan)
        self.assertNotIn(.25, report["pareto"]["frontier_gates"])
        self.assertIsNone(report["pareto"]["operating_point"])


if __name__ == "__main__":
    unittest.main()
