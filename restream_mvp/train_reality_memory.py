"""R0 memory-only SFT; preparation never launches this entry point."""
import argparse
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from restream.reality_dataset import collate_reality
from restream.reality_diagnostics import gradient_norms, probe_subset
from restream.reality_metrics import memory_usage
from restream.training_budget import TrainingBudget
from restream.reality_paired import PairedRealityDataset, paired_loss
from restream.reality_runtime import (read_reality_config, make_memory, make_cache, make_dataset,
                                      prepare_reality, reality_loss, resume_signature)
from restream.runtime import ROOT, load_pipeline


def overfit_indices(dataset, count):
    if count < 4 or count > len(dataset.rows):
        raise ValueError("Overfit set must contain 4..len(dataset) samples")
    groups = {kind: [i for i, row in enumerate(dataset.rows) if row["reference_kind"] == kind]
              for kind in ("async", "aligned", "none", "wrong")}
    if any(not values for values in groups.values()):
        raise ValueError("Overfit check requires all four reference types")
    selected = []
    while len(selected) < count:
        for values in groups.values():
            if values and len(selected) < count:
                selected.append(values.pop(0))
    return selected


def paired_diagnostics_enabled(paired_cfg, step):
    """Sample the per-term gradient decomposition only on steps 1-2 and every
    ``diagnostic_interval`` steps; a long/four-card run should disable it via
    ``diagnostic_gradients: false`` because the extra autograd.grad passes over
    the memory graph are not required for the optimizer."""
    if not paired_cfg.get("diagnostic_gradients", True):
        return False
    interval = paired_cfg.get("diagnostic_interval", 10)
    return step in (1, 2) or step % interval == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--reviewed", action="store_true")
    limits = parser.add_mutually_exclusive_group(required=True)
    limits.add_argument("--max-steps", type=int, help="Legacy budget in batch steps, including empty-memory batches")
    limits.add_argument("--max-updates", type=int, help="Stop after this total number of effective AdamW updates, including resumed updates")
    parser.add_argument("--overfit-samples", type=int, default=0, help="Use a balanced fixed 8–16 sample subset")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/reality_memory_r0")
    parser.add_argument("--probe-overfit", action="store_true", help="Record fixed-noise controls on the full overfit subset before/after training")
    parser.add_argument("--skip-eval", action="store_true", help="Skip final AR evaluation for bounded NCCL/checkpoint smoke only")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("Review the new R0 code/data first; --reviewed explicitly starts training")
    if args.probe_overfit and not args.overfit_samples:
        parser.error("--probe-overfit requires --overfit-samples")
    config = read_reality_config(args.config)
    budget = TrainingBudget(args.max_steps, args.max_updates)
    requested = args.max_steps if args.max_steps is not None else args.max_updates
    limit = config["train"].get("max_updates", config["train"]["max_steps"]) if args.max_updates is not None else config["train"]["max_steps"]
    if requested > limit:
        parser.error(f"Requested budget exceeds configured limit {limit}")
    paired = config["reality_memory"]["objective"].get("paired")
    if paired and (not args.overfit_samples or not args.probe_overfit):
        parser.error("Paired diagnostics require --overfit-samples and --probe-overfit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rank, local, world = (int(os.getenv(name, default)) for name, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    seed = config["seed"]
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    rng = torch.Generator(device=device).manual_seed(seed + rank)
    cache = make_cache(config)
    dataset = make_dataset(config, "train", cache)
    selected = overfit_indices(dataset, args.overfit_samples) if args.overfit_samples else list(range(len(dataset)))
    signature = {**resume_signature(config, cache), "selected_indices": selected, "world_size": world}
    training_dataset = PairedRealityDataset(dataset, paired, selected) if paired else dataset
    subset = Subset(training_dataset, selected)
    sampler = DistributedSampler(subset, world, rank, shuffle=True, seed=seed, drop_last=True)
    loader = DataLoader(subset, batch_size=1, sampler=sampler, collate_fn=collate_reality,
                        num_workers=config["data"]["workers"], pin_memory=True,
                        generator=torch.Generator().manual_seed(seed + rank))
    if not len(loader):
        raise ValueError("Not enough samples for the requested world size")
    pipeline = load_pipeline(config, device)
    memory = make_memory(config).to(device)
    model = DDP(memory, device_ids=[local]) if world > 1 else memory
    opt = torch.optim.AdamW(memory.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    warmup = config["train"]["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1., (step + 1) / max(1, warmup)))
    step, updates, epoch, offset = 0, 0, 0, 0
    if args.resume:
        path = args.resume / "state.pt" if args.resume.is_dir() else args.resume
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume requires identical data, model, reference policy, subset and world size")
        memory.load_state_dict(saved["memory"], strict=True)
        opt.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        step, updates, epoch, offset = saved["step"], saved["optimizer_steps"], saved["epoch"], saved["offset"]
        state = saved["rng"][rank]
        rng.set_state(state["generator"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], device)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
    elif (args.output / "latest.txt").exists():
        raise FileExistsError("Existing run found; use --resume or a new --output directory")
    if budget.done(step, updates):
        raise ValueError("Requested stage must extend the resumed step")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / f"train_rank_{rank}.jsonl"
    print(json.dumps({"rank": rank, "trainable_parameters": sum(p.numel() for p in memory.parameters()),
                      "backbone_frozen": all(not p.requires_grad for p in pipeline.parameters()), "stage": "r0"}), flush=True)
    def probe(phase):
        if paired:
            from restream.reality_paired_diagnostics import probe_paired
            probe_paired(pipeline, memory, dataset, selected, config, device,
                         args.output / f"paired_{phase}_step_{step:04d}_train_rank_{rank}.json", split="train")
            validation = make_dataset(config, "val", cache)
            probe_paired(pipeline, memory, validation, list(range(min(config["eval"]["cases"], len(validation)))),
                         config, device, args.output / f"paired_{phase}_step_{step:04d}_val_rank_{rank}.json", split="val")
        else:
            probe_subset(pipeline, memory, dataset, selected, config, device,
                         args.output / f"probe_{phase}_step_{step:04d}_rank_{rank}.json")
    if args.probe_overfit:
        probe("before")
    idle_batches = 0
    while not budget.done(step, updates):
        sampler.set_epoch(epoch)
        data_started = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < offset:
                data_started = time.perf_counter()
                continue
            data_time = time.perf_counter() - data_started
            started = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            gt, cond, index = prepare_reality(pipeline, batch, device, config)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if paired:
                    loss, stats = paired_loss(pipeline, model, gt, cond, index, batch, rng, config,
                                              diagnostics=paired_diagnostics_enabled(paired, step))
                else:
                    loss, stats = reality_loss(pipeline, model, gt, cond, index, batch["features"].to(device),
                                               batch["reference_mask"].to(device), batch["wrong_reference"].to(device), rng, config)
            loss.backward()
            valid = torch.tensor(int(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
                                     for p in memory.parameters()) and all(p.grad is None for p in pipeline.parameters())), device=device)
            if world > 1:
                dist.all_reduce(valid, op=dist.ReduceOp.MIN)
            if not valid.item():
                raise RuntimeError("Missing/nonfinite memory gradients or unfrozen backbone")
            grouped_grads = gradient_norms(memory)
            norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), config["train"]["grad_clip"])
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite global gradient norm")
            # All-no-memory batches are valid and must not change weights through AdamW decay.
            updated = norm.item() > 0
            if updated:
                opt.step()
                scheduler.step()
                updates += 1
            idle_batches = 0 if updated else idle_batches + 1
            if args.max_updates is not None and idle_batches >= max(16, 2 * len(loader)):
                raise RuntimeError("No effective updates for two epochs (at least 16 batches); stopping an unreachable update budget")
            torch.cuda.synchronize()
            step, offset = step + 1, batch_index + 1
            record = {"rank": rank, "step": step, "optimizer_steps": updates,
                      "batch_step": step, "optimizer_step": updates, "optimizer_updated": updated,
                      "learning_rate": opt.param_groups[0]["lr"], "loss": loss.item(),
                      "video_loss": stats["video_loss"].item(), "grad_norm": norm.item(),
                      **({} if paired else memory_usage(stats, bool(batch["wrong_reference"][0]))),
                      "wrong_source_gate_loss": stats["wrong_loss"].item(), "gradient_norms": grouped_grads,
                      "reference_kind": "paired" if paired else ("wrong_source" if batch["wrong_reference"][0] else batch["reference_kind"][0]),
                      "sample_id": batch["sample_id"][0], "active_memory": stats["active"].any().item(),
                      "data_time": data_time, "compute_time": time.perf_counter() - started,
                      "step_time": time.perf_counter() - started + data_time, "peak_vram_bytes": torch.cuda.max_memory_allocated()}
            if paired:
                paired_fields = {"correct_video_loss": stats["correct_video_loss"].item(),
                                 "wrong_source_video_loss": stats["wrong_source_video_loss"].item(),
                                 "contrast_loss": stats["contrast_loss"].item(),
                                 "correct_memory_gate": stats["gate"][0].item(), "wrong_source_gate": stats["gate"][1].item(),
                                 "correct_relevance_score": stats["relevance_score"][0].item(),
                                 "wrong_source_relevance_score": stats["relevance_score"][1].item(),
                                 "memory_attention_entropy_normalized": stats["attention_entropy_normalized"].tolist(),
                                 "memory_attention_entropy": stats["attention_entropy"].tolist(),
                                 "valid_memory_tokens": stats["valid_memory_tokens"].tolist(),
                                 "prefix_suffix_mse": stats["prefix_suffix_mse"].item(),
                                 "raw_delta_norm": stats["raw_delta_norm"].tolist(),
                                 "applied_delta_norm": stats["applied_delta_norm"].tolist(),
                                 "history_seed": stats["history_seed"], "noise_seed": stats["noise_seed"]}
                for key in ("video_gradient_norm", "contrast_gradient_norm_raw", "contrast_gradient_norm_weighted"):
                    if key in stats:
                        paired_fields[key] = stats[key].item()
                record.update(paired_fields)
            with log_path.open("a") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
            if budget.due(config["train"]["save_every"], step, updates, updated):
                state = {"generator": rng.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
                         "python": random.getstate(), "numpy": np.random.get_state()}
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, state)
                else:
                    states[0] = state
                if rank == 0:
                    folder = args.output / f"step_{step:04d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    torch.save({"stage": "r0", "memory": memory.state_dict(), "optimizer": opt.state_dict(),
                                "scheduler": scheduler.state_dict(), "step": step, "optimizer_steps": updates,
                                "batch_step": step, "optimizer_step": updates, "budget": vars(budget),
                                "epoch": epoch, "offset": offset, "signature": signature, "config": config, "rng": states}, folder / "state.tmp")
                    (folder / "state.tmp").replace(folder / "state.pt")
                    (args.output / "latest.tmp").write_text(str(folder.resolve()) + "\n")
                    (args.output / "latest.tmp").replace(args.output / "latest.txt")
            if not args.skip_eval and not paired and budget.due(config["train"]["eval_every"], step, updates, updated):
                # Every rank evaluates one identical fixed case, avoiding a long idle NCCL wait.
                from eval_reality_memory import evaluate
                evaluate(pipeline, memory, config, device,
                         ROOT / f"outputs/reality_memory/{args.output.name}/step_{step}/rank_{rank}", cases=1, counts=[min(4, config['reality_memory']['references']['max_count'])])
            data_started = time.perf_counter()
            if budget.done(step, updates):
                break
        epoch, offset = epoch + 1, 0
    if args.probe_overfit:
        probe("after")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
