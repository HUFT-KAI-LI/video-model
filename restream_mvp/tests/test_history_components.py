"""D1 history-component protocol and analysis regressions."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream import edit_experiment as ex
from restream.history_component_analysis import analyze
from restream.history_components import CONDITIONS, PROTOCOL, manifest_groups, validate_screen
from scripts import run_history_component_screen as runner


def config():
    return ex.read_config(ROOT / "configs/edit_ready_mvp.yaml")


def plan():
    return json.loads((ROOT / "configs/history_component_screen_plan.json").read_text())


def manifest():
    return json.loads((ROOT / "validation/history_component_screen_manifest.json").read_text())


def synthetic_records():
    effects = {"full_history": 0.1, "global_release": 0.4, "sink_release": 0.5,
               "old_release": 0.2, "recent_release": 0.3, "non_sink_release": 0.35}
    drifts = {"full_history": 0.0, "global_release": 0.2, "sink_release": 0.1,
              "old_release": 0.05, "recent_release": 0.08, "non_sink_release": 0.12}
    records = []
    for entry in manifest()["cases"]:
        condition, chunk = entry["condition"], entry["target_chunk"]
        if chunk == 1:
            p0_hash = "chunk1-release" if condition in ("global_release", "sink_release") else "chunk1-full"
            p1_hash = p0_hash + "-p1"
        else:
            p0_hash, p1_hash = f"chunk4-{condition}-p0", f"chunk4-{condition}-p1"
        effect = effects[condition]
        spec = CONDITIONS[condition]
        records.append({
            "protocol": PROTOCOL, "prompt_id": entry["edit"], "seed": entry["seed"],
            "target_chunk": chunk, "condition": condition,
            "mechanism": "stage_c_global" if spec["kind"] == "global" else "component",
            "history_gate": spec.get("gate"),
            "history_component_gates": copy.deepcopy(spec.get("gates")),
            "evidence": "directional", "D_drift": drifts[condition],
            "cache": {"sha256": f"checkpoint-{entry['edit']}-{chunk}"},
            "chunk_latent_sha256": {"replay": p0_hash, "text_rebind": p1_hash},
            "history_partition": None if condition == "global_release" else {
                policy: {"latent_frames": ({"sink": 3, "old": 0, "recent": 0, "current": 3}
                                           if chunk == 1 else
                                           {"sink": 3, "old": 3, "recent": 3, "current": 3})}
                for policy in ("replay", "text_rebind")},
            "rng": {"replay": {"exact": True}, "text_rebind": {"exact": True}},
            "responsiveness": {"replay": {"S_proxy": 0.0},
                               "text_rebind": {"S_proxy": effect},
                               "full_regeneration": {"S_proxy": 1.0}},
            "editability": {"E": effect, "S_full": 1.0, "R": effect},
            "sanity": {"fixed_references_unmodified_full_history": True,
                       "full_history_P0_exact_base": True if condition == "full_history" else None},
            "preservation": {"outside_exact": True,
                             "identity": {"status": "measured",
                                          "replay": {"mean_cosine": 0.9,
                                                     "mean_cosine_distance": 0.1},
                                          "text_rebind": {"mean_cosine": 0.8,
                                                          "mean_cosine_distance": 0.2}}},
            "boundary": {"replay": {}, "text_rebind": {}},
            "cost": {"status": "invalid_diagnostic"},
        })
    return records


class HistoryComponentTests(unittest.TestCase):
    def test_frozen_manifest_is_exact_48_pair_grid(self):
        groups = manifest_groups(config(), manifest())
        validate_screen(groups, plan())
        self.assertEqual(len(groups), 4)
        self.assertEqual(sum(len(values) for group in groups
                             for values in group["conditions_by_target"].values()), 48)
        self.assertEqual({group["seed"] for group in groups}, {303})

    def test_manifest_rejects_duplicate_and_unknown_condition(self):
        duplicate = manifest()
        duplicate["cases"].append(copy.deepcopy(duplicate["cases"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            manifest_groups(config(), duplicate)
        unknown = manifest()
        unknown["cases"][0]["condition"] = "other"
        with self.assertRaisesRegex(ValueError, "Unknown"):
            manifest_groups(config(), unknown)

    def test_analyzer_reports_selective_dominance_without_significance(self):
        report = analyze(synthetic_records(), plan())
        self.assertTrue(report["invariants_passed"])
        self.assertIn("sink_release", report["pareto"]["frontier_conditions"])
        dominates = {row["condition"]: row["aggregate_dominates_global"]
                     for row in report["pareto"]["selective_dominates_global"]}
        self.assertTrue(dominates["sink_release"])
        self.assertEqual(report["interpretation"],
                         "exploratory_component_screen_no_significance_test")

    def test_analyzer_fails_closed_on_chunk1_router_error(self):
        records = synthetic_records()
        record = next(row for row in records
                      if row["target_chunk"] == 1 and row["condition"] == "recent_release")
        record["chunk_latent_sha256"]["replay"] = "wrong"
        with self.assertRaisesRegex(ValueError, "chunk1 empty recent_release"):
            analyze(records, plan())

    def test_analyzer_fails_closed_on_inert_chunk4_component(self):
        records = synthetic_records()
        unit = [row for row in records if row["prompt_id"] == "dress_red_to_blue"
                and row["target_chunk"] == 4]
        full = next(row for row in unit if row["condition"] == "full_history")
        sink = next(row for row in unit if row["condition"] == "sink_release")
        sink["chunk_latent_sha256"] = copy.deepcopy(full["chunk_latent_sha256"])
        with self.assertRaisesRegex(ValueError, "sink_release is inert"):
            analyze(records, plan())

    def test_analyzer_fails_closed_on_invalid_dino_schema(self):
        records = synthetic_records()
        records[0]["preservation"]["identity"]["replay"] = 0.1
        with self.assertRaisesRegex(ValueError, "invalid DINO"):
            analyze(records, plan())

    def test_runner_reuses_fixed_references_and_restores_context(self):
        class CausalWanSelfAttention(torch.nn.Module):
            pass
        module = CausalWanSelfAttention()
        module.history_gate = .25
        module.history_component_gates = {"sink": .2, "old": .3, "recent": .4}
        pipeline = SimpleNamespace(generator=torch.nn.ModuleList([module]),
                                   num_frame_per_block=3)
        group = manifest_groups(config(), manifest())[0]
        base = torch.full((1, 15, 3, 2, 2), .125)
        full = base + .125
        entries = {chunk: {"path": f"checkpoint-{chunk}", "sha256": f"fixed-{chunk}"}
                   for chunk in (1, 4)}
        generation_calls, replay_calls = [], []

        def generate(unused_pipeline, noise, prompt, **kwargs):
            generation_calls.append((module.history_gate, module.history_component_gates, prompt))
            self.assertEqual(module.history_gate, 1)
            self.assertIsNone(module.history_component_gates)
            return SimpleNamespace(latents=base.clone() if prompt == group["base_prompt"] else full.clone(),
                                   checkpoint_entries=entries)

        def load(entry, *args):
            return SimpleNamespace(target=int(entry["path"].split("-")[-1]),
                                   denoise_noise=[], used=False)

        def replay(unused_pipeline, checkpoint, prompt, **kwargs):
            self.assertFalse(checkpoint.used)
            checkpoint.used = True
            target = checkpoint.target
            if module.history_component_gates is None:
                release = 1.0 - module.history_gate
            else:
                gates = module.history_component_gates
                frame_tokens = 4
                module.last_history_component_tokens = {
                    "sink": 3 * frame_tokens,
                    "old": (0 if target == 1 else 3) * frame_tokens,
                    "recent": (0 if target == 1 else 3) * frame_tokens,
                    "current": 3 * frame_tokens}
                release = 1.0 - gates["sink"]
                if target == 4:
                    release += 2 * (1.0 - gates["old"]) + 3 * (1.0 - gates["recent"])
            replay_calls.append((target, module.history_gate,
                                 copy.deepcopy(module.history_component_gates), prompt))
            value = base[:, target * 3:target * 3 + 3].clone() + release * .125
            if prompt == group["edit_prompt"]:
                value += .0625
            return SimpleNamespace(latents=value, recorded_noise=[])

        def decode(unused_pipeline, latents):
            return torch.cat([latents[:, :1], latents[:, 1:].repeat_interleave(4, dim=1)], dim=1)

        with ExitStack() as stack, tempfile.TemporaryDirectory() as temporary:
            for obj, name, value in (
                (runner.ex, "sample_noise", lambda *args: torch.zeros_like(base)),
                (runner.er, "stream_generate", generate),
                (runner.er, "replay_chunk", replay),
                (runner.er, "decode_latents", decode),
                (runner, "_load_entry", load),
                (runner.emedia, "write_video", lambda *args, **kwargs: None),
                (runner.emedia, "write_comparison", lambda *args, **kwargs: None),
            ):
                stack.enter_context(patch.object(obj, name, value))
            identity = {"model_checkpoint_sha256": "model", "config_hash": "config",
                        "model_identity": {}}
            root = Path(temporary)
            records = runner.run_group(pipeline, config(), group, "cpu", root / "videos",
                                       root / "cache", identity)
        runner.assert_component_invariants(records)
        self.assertEqual(len(records), 12)
        self.assertEqual(len(replay_calls), 24)
        self.assertEqual(len(generation_calls), 2)
        for target in (1, 4):
            self.assertEqual({row["cache"]["sha256"] for row in records
                              if row["target_chunk"] == target}, {f"fixed-{target}"})
        self.assertEqual(module.history_gate, .25)
        self.assertEqual(module.history_component_gates,
                         {"sink": .2, "old": .3, "recent": .4})


if __name__ == "__main__":
    unittest.main()
