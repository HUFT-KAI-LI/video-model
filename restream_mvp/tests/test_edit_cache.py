"""CPU regression tests for the Edit-Ready MVP (Generation-Time Edit Cache).

No GPU and no model weights are required: the checkpoint/restore path is exercised
against a small fake pipeline that has the same attribute surface as
``CausalInferencePipeline``.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream import edit_media as emedia  # noqa: E402
from restream import edit_metrics as em  # noqa: E402
from restream import edit_replay as er  # noqa: E402


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeScheduler:
    def __init__(self):
        self.timesteps = torch.arange(10, dtype=torch.float32)
        self.sigmas = torch.linspace(0, 1, 10)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = torch.nn.Linear(1, 1).to(torch.bfloat16)
        self.freqs = torch.zeros(64, 8)
        self.max_attention_size = 8
        self.local_attn_size = 2


class FakePipeline:
    def __init__(self, blocks=2, cache_len=8, batch=1, block=3, frame_seq_length=4):
        self.num_frame_per_block = block
        self.frame_seq_length = frame_seq_length
        self.local_attn_size = 2
        self.args = SimpleNamespace(model_kwargs=SimpleNamespace(local_attn_size=2, sink_size=1),
                                    context_noise=0)
        self.generator = SimpleNamespace(model=FakeModel())
        self.scheduler = FakeScheduler()
        self.denoising_step_list = torch.tensor([1000, 500, 250, 0])
        self.kv_cache1 = [
            {"k": torch.zeros(batch, cache_len, 2, 4, dtype=torch.bfloat16),
             "v": torch.zeros(batch, cache_len, 2, 4, dtype=torch.bfloat16),
             "global_end_index": torch.tensor([0], dtype=torch.long),
             "local_end_index": torch.tensor([0], dtype=torch.long)}
            for _ in range(blocks)]
        self.crossattn_cache = [
            {"k": torch.zeros(batch, 5, 2, 4, dtype=torch.bfloat16),
             "v": torch.zeros(batch, 5, 2, 4, dtype=torch.bfloat16),
             "is_init": False} for _ in range(blocks)]

    def _set_all_modules_max_attention_size(self, value):
        self.generator.model.max_attention_size = int(value)


def make_checkpoint(pipeline=None, chunk_index=1, device="cpu") -> ec.EditCheckpoint:
    pipeline = pipeline or FakePipeline()
    conditioning = {"prompt_embeds": torch.ones(1, 3, 4)}
    return ec.capture_checkpoint(
        pipeline, sample_id="sample", chunk_index=chunk_index, num_chunks=3,
        prompt="a prompt", seed=7, model_hash="modelhash", model_record={"a": 1},
        config_digest="confighash", conditional_dict=conditioning,
        latent_history=torch.zeros(1, 3, 2, 2, 2), current_start_frame=chunk_index * 3,
        full_latent_shape=(1, 9, 2, 2, 2), next_noise=torch.full((1, 3, 2, 2, 2), 0.5),
        denoise_noise=[torch.ones(3, 2, 2, 2)], device=device)


class CheckpointRoundTrip(unittest.TestCase):
    def test_payload_roundtrip_and_manifest(self):
        pipeline = FakePipeline()
        for index, block in enumerate(pipeline.kv_cache1):
            block["k"].fill_(index + 1)
            block["v"].fill_(index + 2)
            block["global_end_index"].fill_(11)
            block["local_end_index"].fill_(3)
        pipeline.crossattn_cache[0]["is_init"] = True
        checkpoint = make_checkpoint(pipeline)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "chunk001.pt"
            entry = ec.save_edit_checkpoint(path, checkpoint)
            self.assertTrue(path.is_file())
            self.assertEqual(entry["sha256"], ec.sha256_file(path))
            loaded = ec.load_edit_checkpoint(path, verify_sha256=entry["sha256"])
            self.assertEqual(loaded.identity(), checkpoint.identity())
            self.assertTrue(torch.equal(loaded.kv_cache[1]["k"], checkpoint.kv_cache[1]["k"]))
            self.assertEqual(loaded.crossattn_cache[0]["is_init"], True)
            self.assertEqual(loaded.provenance["kv_blocks"], 2)
            self.assertIn("kv_cache[0].k", loaded.provenance["tensors"])
            self.assertEqual(loaded.provenance["cache_bytes"], checkpoint.cache_bytes())

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.pt"
            ec.save_edit_checkpoint(path, make_checkpoint())
            with self.assertRaises(ValueError):
                ec.load_edit_checkpoint(path, verify_sha256="0" * 64)

    def test_schema_and_payload_type_are_validated(self):
        with self.assertRaises(ValueError):
            ec.EditCheckpoint.from_payload({"payload_type": "something-else", "schema_version": 1})
        payload = make_checkpoint().to_payload()
        payload["schema_version"] = 99
        with self.assertRaises(ValueError):
            ec.EditCheckpoint.from_payload(payload)

    def test_identity_verification_rejects_drift(self):
        checkpoint = make_checkpoint()
        ec.verify_checkpoint_identity(checkpoint, model_hash="modelhash", config_digest="confighash",
                                      num_frame_per_block=3, frame_seq_length=4, local_attn_size=2,
                                      latent_shape=(1, 9, 2, 2, 2))
        with self.assertRaises(ValueError):
            ec.verify_checkpoint_identity(checkpoint, model_hash="other")
        with self.assertRaises(ValueError):
            ec.verify_checkpoint_identity(checkpoint, config_digest="other")
        with self.assertRaises(ValueError):
            ec.verify_checkpoint_identity(checkpoint, latent_shape=(1, 6, 2, 2, 2))

    def test_structure_helpers(self):
        structure = {"a": torch.zeros(2, 3), "b": [torch.zeros(4, dtype=torch.int64)]}
        self.assertEqual(ec.structure_bytes(structure), 2 * 3 * 4 + 4 * 8)
        provenance = ec.tensor_provenance(structure)
        self.assertEqual(provenance["a"]["shape"], [2, 3])
        self.assertEqual(provenance["b[0]"]["dtype"], "torch.int64")
        self.assertEqual(ec.canonical_hash({"b": 1, "a": 2}), ec.canonical_hash({"a": 2, "b": 1}))


class RestoreSemantics(unittest.TestCase):
    def test_restore_copies_and_does_not_alias_the_checkpoint(self):
        pipeline = FakePipeline()
        pipeline.kv_cache1[0]["k"].fill_(5)
        checkpoint = make_checkpoint(pipeline)
        restore_target = FakePipeline()
        ec.restore_checkpoint(restore_target, checkpoint, device="cpu")
        self.assertTrue(torch.equal(restore_target.kv_cache1[0]["k"],
                                    torch.full((1, 8, 2, 4), 5.0, dtype=torch.bfloat16)))
        restore_target.kv_cache1[0]["k"].fill_(0)
        self.assertTrue(torch.all(checkpoint.kv_cache[0]["k"] == 5))

    def test_reset_crossattn_clears_text_binding(self):
        pipeline = FakePipeline()
        for block in pipeline.crossattn_cache:
            block["is_init"] = True
            block["k"].fill_(3)
            block["v"].fill_(4)
        checkpoint = make_checkpoint(pipeline)
        target = FakePipeline()
        ec.restore_checkpoint(target, checkpoint, device="cpu", reset_crossattn=True)
        for block in target.crossattn_cache:
            self.assertFalse(block["is_init"])
            self.assertEqual(float(block["k"].abs().sum()), 0.0)
            self.assertEqual(float(block["v"].abs().sum()), 0.0)
        # The saved binding must still be reusable for a same-prompt replay.
        self.assertTrue(torch.all(checkpoint.crossattn_cache[0]["k"] == 3))

    def test_checkpoint_without_kv_cache_is_rejected(self):
        checkpoint = make_checkpoint()
        checkpoint.kv_cache = None
        with self.assertRaises(ValueError):
            ec.restore_checkpoint(FakePipeline(), checkpoint, device="cpu")


class AssemblyPreservation(unittest.TestCase):
    def test_outside_chunks_are_bit_identical(self):
        base = torch.randn(1, 9, 2, 2, 2, dtype=torch.bfloat16)
        edited = torch.randn(1, 3, 2, 2, 2, dtype=torch.bfloat16)
        assembled = er.assemble_latents(base, edited, 1, 3)
        self.assertTrue(torch.equal(assembled[:, :3], base[:, :3]))
        self.assertTrue(torch.equal(assembled[:, 6:], base[:, 6:]))
        self.assertTrue(torch.equal(assembled[:, 3:6], edited))
        self.assertEqual(er.outside_exact(assembled, base, 1, 3), 2)

    def test_tampered_outside_chunk_is_caught(self):
        base = torch.randn(1, 9, 2, 2, 2)
        assembled = base.clone()
        assembled[:, 0, 0, 0, 0] += 1.0
        with self.assertRaises(AssertionError):
            er.outside_exact(assembled, base, 1, 3)

    def test_chunk_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            er.assemble_latents(torch.zeros(1, 9, 2, 2, 2), torch.zeros(1, 2, 2, 2, 2), 1, 3)


class Geometry(unittest.TestCase):
    def test_chunk_slices_tile_the_video_without_gaps(self):
        block, latent_frames = 3, 21
        total = 4 * (latent_frames - 1) + 1
        spans = [er.chunk_frame_slice(index, block, total) for index in range(latent_frames // block)]
        self.assertEqual(spans[0]["pixel_start"], 0)
        self.assertEqual(spans[-1]["pixel_end"], total)
        for left, right in zip(spans, spans[1:]):
            self.assertEqual(left["pixel_end"], right["pixel_start"])
        for span in spans:
            self.assertEqual(span["latent_end"] - span["latent_start"], block)
        self.assertEqual(spans[1]["pixel_start"], 9)
        self.assertEqual(spans[1]["pixel_end"], 21)

    def test_latent_shape_and_noise_sampling(self):
        config = {"generation": {"num_latent_frames": 21, "height": 256, "width": 432, "fps": 16}}
        self.assertEqual(list(er.latent_shape(config)), [1, 21, 16, 32, 54])
        first = ex.sample_noise(config, "cpu", 3)
        second = ex.sample_noise(config, "cpu", 3)
        third = ex.sample_noise(config, "cpu", 4)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, third))
        self.assertEqual(first.dtype, torch.bfloat16)


class Metrics(unittest.TestCase):
    def test_latent_and_pixel_stats(self):
        value = torch.zeros(1, 3, 2, 2, 2)
        self.assertTrue(em.latent_stats(value, value.clone())["exact"])
        other = value.clone()
        other[0, 0, 0, 0, 0] = 0.5
        stats = em.latent_stats(value, other)
        self.assertFalse(stats["exact"])
        self.assertGreater(stats["mse"], 0)
        self.assertAlmostEqual(stats["max_abs"], 0.5, places=6)
        pixels = em.pixel_stats(torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2))
        self.assertEqual(pixels["psnr"], float("inf"))
        self.assertTrue(pixels["exact"])

    def test_solid_colour_fraction(self):
        red = torch.zeros(1, 3, 8, 8)
        red[:, 0] = 1.0
        blue = torch.zeros(1, 3, 8, 8)
        blue[:, 2] = 1.0
        self.assertGreater(em.color_fraction(red, "red"), 0.9)
        self.assertLess(em.color_fraction(red, "blue"), 0.1)
        self.assertGreater(em.color_fraction(blue, "blue"), 0.9)
        black = torch.zeros(1, 3, 8, 8)
        self.assertGreater(em.color_fraction(black, "black"), 0.9)

    def test_colour_probe_reports_direction(self):
        base = torch.zeros(1, 3, 8, 8)
        base[:, 0] = 1.0
        edited = torch.zeros(1, 3, 8, 8)
        edited[:, 2] = 1.0
        probe = {"kind": "color_fraction", "color": "blue", "reference_color": "red"}
        result = em.evaluate_probe(probe, base, edited)
        self.assertGreater(result["S_proxy"], 0.5)
        self.assertIn("delta_target_fraction", result["measurements"])

    def test_luma_probe_direction(self):
        bright = torch.ones(1, 3, 8, 8)
        dark = torch.full((1, 3, 8, 8), 0.2)
        decrease = em.evaluate_probe({"kind": "luma", "direction": "decrease"}, bright, dark)
        increase = em.evaluate_probe({"kind": "luma", "direction": "increase"}, bright, dark)
        self.assertGreater(decrease["S_proxy"], 0)
        self.assertLess(increase["S_proxy"], 0)

    def test_boundary_metrics_and_delta(self):
        latents = torch.zeros(1, 9, 2, 2, 2)
        latents[:, 3:] = 1.0
        values = em.boundary_latent(latents, 1, 3)
        self.assertAlmostEqual(values["left_mse"], 1.0, places=6)
        self.assertAlmostEqual(values["right_mse"], 0.0, places=6)
        delta = em.boundary_delta(values, {"left_mse": 0.5, "right_mse": 0.0})
        self.assertAlmostEqual(delta["delta_left_mse"], 0.5, places=6)

    def test_histogram_distance_between_different_colours(self):
        red = torch.zeros(1, 3, 8, 8)
        red[:, 0] = 1.0
        blue = torch.zeros(1, 3, 8, 8)
        blue[:, 2] = 1.0
        self.assertGreater(em.histogram_distance(red, blue), 0.5)
        self.assertAlmostEqual(em.histogram_distance(red, red.clone()), 0.0, places=6)


class MediaAndExperiment(unittest.TestCase):
    def test_video_writer_roundtrip(self):
        frames = (torch.rand(6, 3, 16, 16) * 255).byte().permute(0, 2, 3, 1).numpy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            emedia.write_video(path, frames, 4.0)
            self.assertTrue(path.is_file() and path.stat().st_size > 0)
            import av
            with av.open(str(path)) as container:
                decoded = sum(1 for _ in container.decode(video=0))
            self.assertEqual(decoded, 6)

    def test_case_expansion_and_sharding(self):
        config = {"seed": 10, "edit": {
            "targets": [1, 4],
            "prompts": [{"id": "a", "base": "x", "edit": "y"},
                        {"id": "b", "base": "x2", "edit": "y2"}]}}
        cases = ex.expand_cases(config)
        self.assertEqual(len(cases), 4)
        self.assertEqual([case["target_chunk"] for case in cases], [1, 4, 1, 4])
        self.assertEqual(cases[0]["seed"], 10)
        self.assertEqual(cases[2]["seed"], 11)
        self.assertEqual(len(ex.shard(cases, 0, 2)), 2)
        self.assertEqual({case["prompt_id"] for case in ex.shard(cases, 1, 2)}, {"a", "b"})
        only = ex.expand_cases(config, prompt_ids=["b"])
        self.assertEqual({case["prompt_id"] for case in only}, {"b"})
        with self.assertRaises(ValueError):
            ex.expand_cases(config, prompt_ids=["missing"])
        with self.assertRaises(ValueError):
            ex.target_chunks(config, num_chunks=2)
        with self.assertRaises(ValueError):
            ex.shard(cases, 3, 2)


class ScriptGates(unittest.TestCase):
    """Regression tests for the gate arithmetic that only runs after a GPU job."""

    @staticmethod
    def _load(name, path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def setUp(self):
        self.runner = self._load("run_chunk_edit", ROOT / "scripts/run_chunk_edit.py")
        self.summarizer = self._load("summarize_edit_ready_mvp",
                                     ROOT / "scripts/summarize_edit_ready_mvp.py")

    @staticmethod
    def _record(local, full, replay=0.0, control=0.0, num_chunks=7, r_time=0.16, outside=True,
                evidence="directional", target_chunk=1, recache_exact=True,
                control_exact=True, recache_eq_rebind=True, r_e2e=0.4):
        return {
            "sample_id": "case", "prompt_id": "p", "target_chunk": target_chunk,
            "num_chunks": num_chunks, "evidence": evidence,
            "responsiveness": {"text_rebind": {"S_proxy": local},
                               "replay": {"S_proxy": replay},
                               "crossattn_control": {"S_proxy": control},
                               "full_regeneration": {"S_proxy": full}},
            "preservation": {"outside_exact": outside, "chunks_checked": num_chunks - 1,
                             "outside_max_abs_after_vae": 0.01},
            "cost": {"R_time_generation": r_time, "R_time_compute": r_time * 0.9,
                     "R_time_end_to_end": r_e2e,
                     "partial_generation_seconds": 0.5, "full_generation_seconds": 3.2,
                     "partial_end_to_end_seconds": 0.9, "disk_load_seconds": 0.1,
                     "host_to_device_seconds": 0.05, "restore_seconds": 0.02,
                     "encode_seconds": 0.03, "denoise_seconds": 0.3,
                     "cache_bytes_per_chunk": 1049887200, "cache_disk_bytes": 1055165467,
                     "peak_vram_bytes": 1, "page_cache_dropped": True},
            "editability": {"chunk0_matches_full_regeneration":
                            True if target_chunk == 0 else None},
            "sanity": {"control_equals_replay": control_exact,
                       "recache_equals_text_rebind": recache_eq_rebind},
            "recache_cache_matches_checkpoint": {"comparable": True, "exact": recache_exact},
            "boundary": {"latent": {"base": {"left_mse": 0.01, "right_mse": 0.02},
                                    "edited": {"left_mse": 0.011, "right_mse": 0.021}}},
        }

    def test_runner_gates_strong_and_weak_cases(self):
        config = {"gates": {"edit": {"min_s_proxy": 0.0, "min_full_regeneration_ratio": 0.25},
                            "cost": {"max_time_ratio": 0.5, "min_chunks": 5}}}
        strong = self.runner.evaluate_gates([self._record(local=0.10, full=0.20)], config)
        self.assertTrue(strong["gate_b_editability"]["passed"])
        self.assertEqual(strong["gate_b_editability"]["strong_cases"], 1)
        weak = self.runner.evaluate_gates([self._record(local=0.001, full=0.20)], config)
        self.assertTrue(weak["gate_b_editability"]["passed"])
        self.assertEqual(weak["gate_b_editability"]["strong_cases"], 0)
        failed = self.runner.evaluate_gates([self._record(local=0.0, full=0.20)], config)
        self.assertFalse(failed["gate_b_editability"]["passed"])
        self.assertFalse(self.runner.evaluate_gates([self._record(local=0.1, full=0.2, outside=False)],
                                                    config)["gate_c_preservation"]["passed"])
        slow = self.runner.evaluate_gates([self._record(local=0.1, full=0.2, r_e2e=0.9)], config)
        self.assertFalse(slow["gate_d_cost"]["passed"])

    def test_qualitative_cases_are_excluded_from_gate_b(self):
        config = {"gates": {"edit": {"min_s_proxy": 0.0, "min_full_regeneration_ratio": 0.25},
                            "cost": {"max_time_ratio": 0.5, "min_chunks": 5}}}
        gates = self.runner.evaluate_gates(
            [self._record(local=0.9, full=1.0, evidence="qualitative")], config)
        edit_gate = gates["gate_b_editability"]
        self.assertFalse(edit_gate["passed"])          # no directional case at all
        self.assertEqual(edit_gate["semantic_cases"], 0)
        self.assertEqual(edit_gate["qualitative_cases"], 1)
        self.assertEqual(edit_gate["qualitative_consistent_cases"], 1)

    def test_r_k_by_chunk_and_sanity_counters(self):
        config = {"gates": {"edit": {"min_s_proxy": 0.0, "min_full_regeneration_ratio": 0.25},
                            "cost": {"max_time_ratio": 0.5, "min_chunks": 5}}}
        records = [self._record(local=0.2, full=0.2, target_chunk=0),
                   self._record(local=0.02, full=0.2, target_chunk=1),
                   self._record(local=0.004, full=0.2, target_chunk=4)]
        gates = self.runner.evaluate_gates(records, config)
        by_chunk = gates["gate_b_editability"]["R_k_by_chunk"]
        self.assertAlmostEqual(by_chunk["0"][0], 1.0)
        self.assertAlmostEqual(by_chunk["1"][0], 0.1)
        self.assertAlmostEqual(by_chunk["4"][0], 0.02)

    def test_chunk0_control_is_not_informative(self):
        config = {"gates": {"edit": {"min_s_proxy": 0.0, "min_full_regeneration_ratio": 0.25},
                            "cost": {"max_time_ratio": 0.5, "min_chunks": 5}}}
        # At chunk 0 the cached text K/V is empty, so control == edit; that must
        # not veto an otherwise directional case.
        gates = self.runner.evaluate_gates(
            [self._record(local=0.1, full=0.1, control=0.1, target_chunk=0)], config)
        self.assertTrue(gates["gate_b_editability"]["passed"])
        self.assertFalse(gates["gate_b_editability"]["cases"][0]["control_informative"])
        same = self.runner.evaluate_gates(
            [self._record(local=0.1, full=0.1, control=0.1, target_chunk=1)], config)
        self.assertFalse(same["gate_b_editability"]["passed"])
        cases = self.summarizer.gate_b_cases(
            [self._record(local=0.1, full=0.1, control=0.1, target_chunk=0)], 0.0, 0.25)
        self.assertTrue(cases[0]["passed"])

    def test_summarizer_boundary_and_decision(self):
        config = {"max_relative_increase": 1.0, "absolute_slack": 1e-4}
        good = self.summarizer.boundary_verdict([self._record(0.1, 0.2)], config)
        self.assertTrue(good["left"]["acceptable"] and good["right"]["acceptable"])
        bad = self.summarizer.boundary_verdict(
            [{"boundary": {"latent": {"base": {"left_mse": 1e-6, "right_mse": 1e-6},
                                      "edited": {"left_mse": 0.5, "right_mse": 0.5}}}}], config)
        self.assertFalse(bad["left"]["acceptable"])
        self.assertEqual(self.summarizer.decision(False, True, 1, 4, bad, {}),
                         "NO_GO_STATE_INCOMPLETE")
        self.assertEqual(self.summarizer.decision(True, False, 0, 0, bad, {}),
                         "NO_GO_NEED_PROMPT_REBINDING")
        self.assertEqual(self.summarizer.decision(True, True, 0, 5, bad, {}),
                         "GO_WEAK_PROMPT_REBINDING")
        # One strong case out of 23 is not enough to claim a strong GO.
        self.assertEqual(self.summarizer.decision(True, True, 1, 23, good, {}),
                         "GO_WEAK_PROMPT_REBINDING")
        self.assertEqual(self.summarizer.decision(True, True, 2, 4, good, {}), "STRONG_GO")

    def test_summarizer_splits_semantic_and_qualitative(self):
        records = [self._record(local=0.1, full=0.2, evidence="directional"),
                   self._record(local=0.5, full=0.6, evidence="qualitative")]
        cases = self.summarizer.gate_b_cases(records, 0.0, 0.25)
        semantic = [case for case in cases if case["evidence"] == "directional"]
        qualitative = [case for case in cases if case["evidence"] != "directional"]
        self.assertEqual(len(semantic), 1)
        self.assertEqual(len(qualitative), 1)
        self.assertTrue(semantic[0]["passed"])
        self.assertTrue(semantic[0]["strong"])

    def test_provenance_records_cleanliness_and_tree_digest(self):
        state = ec.git_state()
        self.assertIn("git_commit", state)
        self.assertIn("git_dirty", state)
        self.assertIn("git_status_sha256", state)
        digest = ec.source_tree_sha256()
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, ec.source_tree_sha256())
        self.assertNotEqual(digest, ec.source_tree_sha256(prefixes=("nonexistent/",)))


class RecacheAndCaches(unittest.TestCase):
    def test_compare_caches_detects_exact_and_drift(self):
        left = [{"k": torch.zeros(2, 3), "v": torch.ones(2, 3),
                 "global_end_index": torch.tensor([5]), "local_end_index": torch.tensor([2])}]
        right = [{"k": torch.zeros(2, 3), "v": torch.ones(2, 3),
                  "global_end_index": torch.tensor([5]), "local_end_index": torch.tensor([2])}]
        self.assertTrue(er.compare_caches(left, right)["exact"])
        right[0]["k"][0, 0] = 1.0
        result = er.compare_caches(left, right)
        self.assertFalse(result["exact"])
        self.assertAlmostEqual(result["max_abs_diff"], 1.0, places=6)
        self.assertFalse(er.compare_caches(left, [])["comparable"])

    def test_replay_rejects_foreign_checkpoint_identity(self):
        pipeline = FakePipeline()
        checkpoint = make_checkpoint(pipeline)
        with self.assertRaises(ValueError):
            er.replay_chunk(pipeline, checkpoint, "a prompt", device="cpu",
                            expected_model_hash="someone-elses-model", time_it=False)
        with self.assertRaises(ValueError):
            er.replay_chunk(pipeline, checkpoint, "a prompt", device="cpu",
                            expected_config_hash="another-config", time_it=False)

    def test_replay_rejects_geometry_mismatch(self):
        pipeline = FakePipeline()
        checkpoint = make_checkpoint(pipeline)
        other = FakePipeline(frame_seq_length=8)
        with self.assertRaises(ValueError):
            er.replay_chunk(other, checkpoint, "a prompt", device="cpu", time_it=False)

    def test_drop_page_cache_is_scoped_and_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"x" * 4096)
            ex.drop_page_cache(path)  # must not raise even when unsupported
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
