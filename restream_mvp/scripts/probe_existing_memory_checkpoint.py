"""Counterfactual teacher-forcing probes on an existing paired checkpoint.

Loads ONLY the memory ``state_dict`` of an ``r0`` paired checkpoint. It never
restores the optimizer, scheduler, RNG or training offset, so a checkpoint
produced under the previous review round can be probed with the current
controls (No Memory / Active Zero / Global Mean / Pair Mean / Correct / Wrong
Source) without retraining and without pretending the new controls changed the
weights.

Strict checks before probing:
  * checkpoint stage is r0 with a paired objective;
  * checkpoint signature matches its own recorded config (self-consistency);
  * encoder (cache) identity and train/val manifest digests match the current
    config, so weights are probed on the data they were trained with;
  * the current config differs from the checkpoint config only in
    evaluation-only fields (probe seeds, global-constant file, diagnostics,
    train budgets, eval cases, random seed). Architecture, data, reference
    policy, objective and regularization semantics must be identical.

The script performs no optimizer update: ``torch.autograd.grad`` and backward
are never called on the memory parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json
from restream.reality_paired import validate_paired_config
from restream.reality_runtime import (read_reality_config, make_cache, make_dataset,
                                      make_memory, resume_signature)
from restream.reality_selection import normalize_selection_defaults, select_targets
from restream.runtime import ROOT, load_pipeline
from train_reality_memory import overfit_indices

# Fields whose difference is evaluation-only and therefore allowed while probing
# existing weights. Anything outside this list must match the checkpoint. Both
# configs are normalized with the pre-identity defaults first, so an older
# checkpoint that predates these keys is compared on its effective semantics.
ALLOWED_EVAL_PREFIXES = (
    ("seed",),
    ("train",),
    ("eval",),
    ("reality_memory", "references", "global_constant_features"),
    ("reality_memory", "references", "allow_legacy_offline_manifest"),
    ("reality_memory", "objective", "paired", "probe_noise_seeds"),
    ("reality_memory", "objective", "paired", "diagnostic_gradients"),
    ("reality_memory", "objective", "paired", "diagnostic_interval"),
)


def diff_configs(left, right, path=()):
    """List (path, reason) differences that are not covered by ALLOWED_EVAL_PREFIXES."""
    problems = []
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left:
                problems.append((path + (key,), "present in probe config but missing in checkpoint"))
            elif key not in right:
                problems.append((path + (key,), "present in checkpoint but missing in probe config"))
            else:
                problems.extend(diff_configs(left[key], right[key], path + (key,)))
    elif left != right:
        problems.append((path, left, right))
    return [problem for problem in problems
            if not any(problem[0][:len(prefix)] == prefix for prefix in ALLOWED_EVAL_PREFIXES)]


def verify_checkpoint(saved, config, cache):
    """Raise unless the checkpoint is an r0 paired run that is self-consistent and
    semantically compatible with the probe config (beyond evaluation-only fields).

    The stored signature also carries ``selected_indices`` and ``world_size``, so
    only the semantic fields shared with ``resume_signature`` are compared.
    """
    if saved.get("stage") != "r0" or "paired" not in (saved.get("config", {}).get("reality_memory", {}).get("objective") or {}):
        raise ValueError("Checkpoint is not an r0 paired run")
    recorded = saved.get("signature") or {}
    recomputed = resume_signature(saved["config"], cache)
    for key in ("config", "encoder", "manifests"):
        if recorded.get(key) != recomputed[key]:
            raise ValueError(f"Checkpoint signature field {key!r} does not match its own recorded config")
    expected = resume_signature(config, cache)
    if recorded.get("encoder") != expected["encoder"] or recorded.get("manifests") != expected["manifests"]:
        raise ValueError("Checkpoint encoder identity or train/val manifest digest differs from the probe config")
    problems = diff_configs(normalize_selection_defaults(saved["config"]), normalize_selection_defaults(config))
    if problems:
        raise ValueError(f"Checkpoint semantics differ beyond evaluation-only fields: {problems}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_paired.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory containing state.pt, or the state.pt file")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--cases", type=int, default=-1, help="Probe targets: -1 = eval.cases, 0 = all rows, N = seeded sample of N unique targets")
    parser.add_argument("--target-seed", type=int, default=None, help="Seed for deterministic target selection (default: config seed)")
    parser.add_argument("--overfit-samples", type=int, default=0, help="Required for --split train: balanced fixed subset like training probes")
    parser.add_argument("--noise-seeds", type=int, nargs="+", help="Override probe_noise_seeds (evaluation-only; e.g. 3-5 seeds per target)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reality_memory/probe_existing")
    args = parser.parse_args()
    if args.split == "train" and not args.overfit_samples:
        parser.error("--split train requires --overfit-samples (training probes used the balanced fixed subset)")
    checkpoint = args.checkpoint / "state.pt" if args.checkpoint.is_dir() else args.checkpoint
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = read_reality_config(args.config)
    if args.noise_seeds:
        seeds = list(args.noise_seeds)
        if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
            parser.error("--noise-seeds must be distinct nonnegative integers")
        config["reality_memory"]["objective"]["paired"]["probe_noise_seeds"] = seeds
        validate_paired_config(config)
    cache = make_cache(config)
    verify_checkpoint(saved, config, cache)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    pipeline = load_pipeline(config, device)
    memory = make_memory(config).to(device)
    memory.load_state_dict(saved["memory"], strict=True)
    dataset = make_dataset(config, args.split, cache)
    target_seed = config["seed"] if args.target_seed is None else args.target_seed
    if args.split == "train":
        indices = overfit_indices(dataset, args.overfit_samples)
        selection = {"mode": "balanced_overfit_subset", "count": len(indices), "seed": config["seed"]}
    else:
        count = config["eval"]["cases"] if args.cases < 0 else args.cases
        indices = select_targets(dataset.rows, count, target_seed)
        selection = {"mode": "all" if count <= 0 or count >= len(dataset.rows) else "seeded_sample",
                     "count": len(indices), "seed": target_seed}
    from restream.reality_paired_diagnostics import probe_paired
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = probe_paired(pipeline, memory, dataset, indices, config, device,
                          output / f"paired_existing_{args.split}.json", split=args.split)
    verification = {"checkpoint": {"path": str(checkpoint.resolve()), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                    "memory_only_load": True,
                    "optimizer_scheduler_rng_restored": False,
                    "stage": saved["stage"], "batch_step": saved["batch_step"], "optimizer_step": saved["optimizer_step"],
                    "signature_self_consistent": True,
                    "encoder_identity_matches": True, "manifests_match": True,
                    "semantics_equal_beyond_eval_only": True, "split": args.split,
                    "probe_noise_seeds": config["reality_memory"]["objective"]["paired"]["probe_noise_seeds"],
                    "legacy_defaults_normalized": True,
                    "targets_probed": len(indices), "target_selection": selection,
                    "target_sample_ids": [dataset.rows[index]["sample_id"] for index in indices],
                    "no_parameter_update": True,
                    "global_constant": report["global_constant"],
                    "notes": ["Probes load the memory state only; optimizer/scheduler/RNG/training offset are not restored.",
                              "Controls: No Memory, Active Zero, Global Mean (positive/async/aligned role means), Pair Mean, Correct, Wrong Source; no backward is executed.",
                              "Configs are compared after filling pre-identity defaults, so a checkpoint that predates selection_seed/temporal_sampling/pool_size is checked on its effective semantics."]}
    write_json(output / "probe_existing_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()