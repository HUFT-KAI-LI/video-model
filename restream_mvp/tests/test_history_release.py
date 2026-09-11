"""CPU protocol regressions. These do not substitute for sealed CUDA smoke."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import unittest
import tempfile
from unittest.mock import patch
from contextlib import ExitStack
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.history_release import manifest_groups, validate_smoke, load_sealed
from restream import edit_experiment as ex

spec = importlib.util.spec_from_file_location("paired_runner", ROOT / "scripts/run_chunk_edit.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def config():
    return ex.read_config(ROOT / "configs/edit_ready_mvp.yaml")


def smoke():
    return json.loads((ROOT / "validation/history_release_smoke_manifest.json").read_text())


def check_manifest_uses_every_explicit_seed_and_sparse_combination():
    manifest = json.loads((ROOT / "validation/history_release_sweep_manifest.json").read_text())
    groups = manifest_groups(config(), manifest)
    assert len(groups) == 8
    assert all({g["seed"] for g in groups if g["prompt_id"] == edit} == {42, 43}
               for edit in {g["prompt_id"] for g in groups})
    assert sum(sum(len(v) for v in g["gates_by_target"].values()) for g in groups) == 120
    sparse = {"schema": 2, "cases": [manifest["cases"][0], manifest["cases"][-1]]}
    assert sum(len(g["targets"]) for g in manifest_groups(config(), sparse)) == 2
    assert sum(sum(len(v) for v in g["gates_by_target"].values())
               for g in manifest_groups(config(), sparse)) == 2


def check_manifest_rejects_ambiguous_or_invalid_cases(mutation):
    manifest = smoke()
    if mutation == "policy": manifest["cases"][0]["policy"] = "P0"
    if mutation == "duplicate": manifest["cases"].append(manifest["cases"][0])
    if mutation == "nan": manifest["cases"][0]["history_gate"] = float("nan")
    if mutation == "negative_target": manifest["cases"][0]["target_chunk"] = -1
    if mutation == "schema": manifest["schema"] = 1
    if mutation == "unknown": manifest["cases"][0]["edit"] = "unknown"
    with unittest.TestCase().assertRaises(ValueError): manifest_groups(config(), manifest)


def check_drift_is_subtracted_and_timing_never_passes():
    response = {"replay": {"S_proxy": .6}, "text_rebind": {"S_proxy": .5},
                "full_regeneration": {"S_proxy": 1.0}}
    record = {"protocol": "fixed_history_paired_v2", "sample_id": "x", "prompt_id": "x",
              "seed": 42, "target_chunk": 1, "history_gate": .5, "D_drift": .2,
              "responsiveness": response, "boundary": {},
              "preservation": {"outside_exact": True}, "evidence": "directional"}
    scores = runner.paired_editability(response)
    assert abs(scores["E"] + .1) < 1e-12
    assert abs(scores["R_k"] + .1) < 1e-12
    for gate in (0, .5, 1):
        record["history_gate"] = gate
        report = runner.evaluate_gates([record], config())
        assert not report["gate_b_editability"]["passed"]
        assert report["gate_b_editability"]["status"] == \
            "descriptive_only_requires_frozen_analysis"
        assert report["gate_d_cost"]["passed"] is None
        assert report["gate_d_cost"]["counted_cases"] == 0
    response["full_regeneration"]["S_proxy"] = 0
    assert runner.paired_editability(response)["R_k"] is None


def setup_fake(monkeypatch, tmp_path):
    class CausalWanSelfAttention(torch.nn.Module):
        pass
    module = CausalWanSelfAttention()
    module.history_gate = .25  # caller state must be restored after the group
    pipeline = SimpleNamespace(generator=torch.nn.ModuleList([module]), num_frame_per_block=3)
    group = manifest_groups(config(), smoke())[0]
    base = torch.full((1, 15, 3, 2, 2), .125)
    full = base + .125
    generation_calls, replay_calls, load_calls = [], [], []
    entries = {k: {"path": f"checkpoint-{k}", "sha256": f"immutable-{k}"} for k in [0, 1, 4]}
    def generate(pipeline, noise, prompt, **kwargs):
        generation_calls.append((module.history_gate, prompt, kwargs))
        assert module.history_gate == 1
        return SimpleNamespace(latents=base.clone() if prompt == group["base_prompt"] else full.clone(),
                               checkpoint_entries=entries)
    def load(entry, *args):
        load_calls.append(dict(entry))
        return SimpleNamespace(target=int(entry["path"].split("-")[-1]), denoise_noise=[], used=False)
    def replay(pipeline, checkpoint, prompt, **kwargs):
        assert not checkpoint.used  # both policies must get a fresh checkpoint
        checkpoint.used = True
        target = checkpoint.target
        replay_calls.append((target, module.history_gate, prompt, kwargs))
        value = base[:, target * 3:target * 3 + 3].clone()
        if prompt == group["edit_prompt"]:
            value += .125 if target == 0 else .0625
        if target > 0: value += (1 - module.history_gate) * .25
        return SimpleNamespace(latents=value, recorded_noise=[])
    def decode(pipeline, latents):
        return torch.cat([latents[:, :1], latents[:, 1:].repeat_interleave(4, dim=1)], dim=1)
    monkeypatch.setattr(runner.ex, "sample_noise", lambda *args: torch.zeros_like(base))
    monkeypatch.setattr(runner.er, "stream_generate", generate)
    monkeypatch.setattr(runner.er, "replay_chunk", replay)
    monkeypatch.setattr(runner.er, "decode_latents", decode)
    monkeypatch.setattr(runner, "_load_entry", load)
    monkeypatch.setattr(runner.emedia, "write_video", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.emedia, "write_comparison", lambda *args, **kwargs: None)
    identity = {"model_checkpoint_sha256": "model", "config_hash": "config", "model_identity": {}}
    return pipeline, group, identity, generation_calls, replay_calls, load_calls


def check_runner_fixed_history_18_replays_and_sealed_check(monkeypatch, tmp_path):
    pipeline, group, identity, generations, replays, loads = setup_fake(monkeypatch, tmp_path)
    validate_smoke([group])
    records = runner.run_group(pipeline, config(), group, "cpu", tmp_path / "first",
                               tmp_path / "cache", identity)
    assert len(records) == 9 and len(replays) == 18 and len(generations) == 2
    assert pipeline.generator[0].history_gate == .25
    for target in (0, 1, 4):
        assert len({r["cache"]["sha256"] for r in records if r["target_chunk"] == target}) == 1
        assert sum(e["path"] == f"checkpoint-{target}" for e in loads) == 6
    assert all(r["D_drift"] == 0 for r in records if r["history_gate"] == 1 or r["target_chunk"] == 0)
    assert any(r["D_drift"] > 0 for r in records if r["history_gate"] < 1 and r["target_chunk"] > 0)
    assert len({r["sample_id"] for r in records}) == 9
    assert all(r["preservation"]["outside_exact"] for r in records)
    sealed = {(r["prompt_id"], r["seed"], r["target_chunk"]): r for r in records if r["history_gate"] == 1}
    second = runner.run_group(pipeline, config(), group, "cpu", tmp_path / "second",
                              tmp_path / "cache", identity, sealed=sealed)
    assert all(r["sanity"]["g1_P1_exact_sealed"] for r in second if r["history_gate"] == 1)
    # A changed sealed result must abort the run and restore caller module gates.
    sealed[(group["prompt_id"], 42, 0)]["chunk_latent_sha256"]["text_rebind"] = "wrong"
    with unittest.TestCase().assertRaisesRegex(AssertionError, "sealed text_rebind"):
        runner.run_group(pipeline, config(), group, "cpu", tmp_path / "bad",
                         tmp_path / "cache", identity, sealed=sealed)
    assert pipeline.generator[0].history_gate == .25


def check_sealed_loader_checks_coverage_and_identity(tmp_path):
    groups = manifest_groups(config(), smoke())
    group = groups[0]
    identity = {"config_hash": "c", "model_checkpoint_sha256": "m"}
    cases = [{"prompt_id": group["prompt_id"], "seed": 42, "target_chunk": k,
              "base_prompt": group["base_prompt"], "edit_prompt": group["edit_prompt"],
              "chunk_latent_sha256": {v: "digest" for v in ("original", "text_rebind", "full_regeneration")}}
             for k in (0, 1, 4)]
    path = tmp_path / "sealed.json"
    path.write_text(json.dumps({"provenance": identity, "cases": cases}))
    assert len(load_sealed([path], groups, identity)[0]) == 3
    with unittest.TestCase().assertRaisesRegex(ValueError, "config_hash"):
        load_sealed([path], groups, dict(identity, config_hash="other"))
    path.write_text(json.dumps({"provenance": identity, "cases": cases[:1]}))
    with unittest.TestCase().assertRaisesRegex(ValueError, "Missing sealed"):
        load_sealed([path], groups, identity)
    partial, _ = load_sealed([path], groups, identity, require_all=False)
    assert len(partial) == 1


class HistoryReleaseTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp_path = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.monkeypatch = SimpleNamespace(
            setattr=lambda obj, name, value: self.stack.enter_context(patch.object(obj, name, value)))

    def test_manifest(self):
        check_manifest_uses_every_explicit_seed_and_sparse_combination()

    def test_invalid_manifests(self):
        for mutation in ("policy", "duplicate", "nan", "negative_target", "schema", "unknown"):
            with self.subTest(mutation=mutation):
                check_manifest_rejects_ambiguous_or_invalid_cases(mutation)

    def test_metrics(self):
        check_drift_is_subtracted_and_timing_never_passes()

    def test_inert_gate_fails_closed(self):
        base = {"target_chunk": 1, "chunk_latent_sha256":
                {"replay": "same-p0", "text_rebind": "same-p1"}}
        inert = [dict(base, history_gate=gate) for gate in (1, .5, 0)]
        with self.assertRaisesRegex(AssertionError, "intervention is inert"):
            runner.assert_gate_intervention_active(inert)
        active = [dict(base, history_gate=1),
                  {**base, "history_gate": 0,
                   "chunk_latent_sha256": {"replay": "changed", "text_rebind": "same-p1"}}]
        runner.assert_gate_intervention_active(active)

    def test_runner(self):
        check_runner_fixed_history_18_replays_and_sealed_check(self.monkeypatch, self.tmp_path)

    def test_sealed(self):
        check_sealed_loader_checks_coverage_and_identity(self.tmp_path)

    def test_smoke_invariants_fail_closed(self):
        for failure in ("g1_p0", "chunk0_p1", "outside"):
            with self.subTest(failure=failure):
                pipeline, group, identity, *_ = setup_fake(self.monkeypatch, self.tmp_path)
                replay = runner.er.replay_chunk
                assemble = runner.er.assemble_latents
                def broken_replay(pipeline, checkpoint, prompt, **kwargs):
                    result = replay(pipeline, checkpoint, prompt, **kwargs)
                    gate = pipeline.generator[0].history_gate
                    if ((failure == "g1_p0" and gate == 1 and prompt == group["base_prompt"])
                        or (failure == "chunk0_p1" and gate == .5 and prompt == group["edit_prompt"])):
                        result.latents += .25
                    return result
                def broken_assemble(*args):
                    value = assemble(*args)
                    if failure == "outside":
                        value[:, -1] += 1
                    return value
                with patch.object(runner.er, "replay_chunk", broken_replay), \
                     patch.object(runner.er, "assemble_latents", broken_assemble):
                    with self.assertRaises(AssertionError):
                        runner.run_group(pipeline, config(), group, "cpu",
                                         self.tmp_path / failure, self.tmp_path / "cache", identity)
                self.assertEqual(pipeline.generator[0].history_gate, .25)
