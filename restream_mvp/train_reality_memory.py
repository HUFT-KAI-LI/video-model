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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--max-steps", type=int, required=True, help="Explicit short stage budget; begin with 10")
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
    if not 1 <= args.max_steps <= config["train"]["max_steps"]:
        parser.error("--max-steps must be positive and within the configured 500-step R0 limit")
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
    subset = Subset(dataset, selected)
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
    if step >= args.max_steps:
        raise ValueError("Requested stage must extend the resumed step")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / f"train_rank_{rank}.jsonl"
    print(json.dumps({"rank": rank, "trainable_parameters": sum(p.numel() for p in memory.parameters()),
                      "backbone_frozen": all(not p.requires_grad for p in pipeline.parameters()), "stage": "r0"}), flush=True)
    if args.probe_overfit:
        probe_subset(pipeline, memory, dataset, selected, config, device,
                     args.output / f"probe_before_step_{step:04d}_rank_{rank}.json")
    while step < args.max_steps:
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
            if norm.item() > 0:
                opt.step()
                scheduler.step()
                updates += 1
            torch.cuda.synchronize()
            step, offset = step + 1, batch_index + 1
            record = {"rank": rank, "step": step, "optimizer_steps": updates, "loss": loss.item(),
                      "video_loss": stats["video_loss"].item(), "grad_norm": norm.item(),
                      **memory_usage(stats, bool(batch["wrong_reference"][0])),
                      "wrong_source_gate_loss": stats["wrong_loss"].item(), "gradient_norms": grouped_grads,
                      "reference_kind": "wrong_source" if batch["wrong_reference"][0] else batch["reference_kind"][0],
                      "sample_id": batch["sample_id"][0], "active_memory": stats["active"].any().item(),
                      "data_time": data_time, "compute_time": time.perf_counter() - started,
                      "step_time": time.perf_counter() - started + data_time, "peak_vram_bytes": torch.cuda.max_memory_allocated()}
            with log_path.open("a") as log:
                log.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record), flush=True)
            if step % config["train"]["save_every"] == 0 or step == args.max_steps:
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
                                "epoch": epoch, "offset": offset, "signature": signature, "config": config, "rng": states}, folder / "state.tmp")
                    (folder / "state.tmp").replace(folder / "state.pt")
                    (args.output / "latest.tmp").write_text(str(folder.resolve()) + "\n")
                    (args.output / "latest.tmp").replace(args.output / "latest.txt")
            if not args.skip_eval and (step % config["train"]["eval_every"] == 0 or step == args.max_steps):
                # Every rank evaluates one identical fixed case, avoiding a long idle NCCL wait.
                from eval_reality_memory import evaluate
                evaluate(pipeline, memory, config, device,
                         ROOT / f"outputs/reality_memory/{args.output.name}/step_{step}/rank_{rank}", cases=1, counts=[min(4, config['reality_memory']['references']['max_count'])])
            data_started = time.perf_counter()
            if step >= args.max_steps:
                break
        epoch, offset = epoch + 1, 0
    if args.probe_overfit:
        probe_subset(pipeline, memory, dataset, selected, config, device,
                     args.output / f"probe_after_step_{step:04d}_rank_{rank}.json")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
