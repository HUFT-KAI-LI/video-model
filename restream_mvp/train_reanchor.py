"""Adapter-only SFT. Launch only after human review (never run by prepare)."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from restream.anchor_adapter import GatedAnchorAdapter
from restream.dataset import VideoDataset
from restream.objective import prepare, future_loss
from restream.runtime import ROOT, read_config, load_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/restream_mvp.yaml"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--reviewed", action="store_true", help="Explicit post-review experiment launch")
    a = p.parse_args()
    if not a.reviewed:
        p.error("Preparation stops before training; use --reviewed only after reviewing the code/data")
    c = read_config(a.config)
    if a.max_steps:
        c["train"]["max_steps"] = a.max_steps
    rank, local, world = int(os.getenv("RANK", 0)), int(os.getenv("LOCAL_RANK", 0)), int(os.getenv("WORLD_SIZE", 1))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl")
    seed = c["seed"]
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)  # identical initial adapter on all ranks
    rng = torch.Generator(device=device).manual_seed(seed + rank)
    dataset = VideoDataset(ROOT / c["data"]["train_manifest"], c["data"]["frames"], c["data"]["height"], c["data"]["width"])
    manifest_sha256 = hashlib.sha256((ROOT / c["data"]["train_manifest"]).read_bytes()).hexdigest()
    sampler = DistributedSampler(dataset, world, rank, shuffle=True, seed=seed, drop_last=True)
    workers = c["data"]["workers"]
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=workers, pin_memory=True,
                        **({"persistent_workers": True, "prefetch_factor": 2} if workers else {}))
    if not len(loader):
        raise ValueError("Insufficient training data for the requested world size")
    pipeline = load_pipeline(c, device)
    adapter = GatedAnchorAdapter(c["reanchor"]["channels"], c["reanchor"]["gate_init_logit"]).to(device)
    model = DDP(adapter, device_ids=[local]) if world > 1 else adapter
    opt = torch.optim.AdamW(adapter.parameters(), lr=c["train"]["lr"], weight_decay=c["train"]["weight_decay"])
    warmup = c["train"]["warmup_steps"]
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1., (step + 1) / max(1, warmup)))
    step, epoch, offset = 0, 0, 0
    if a.resume:
        path = a.resume / "state.pt" if a.resume.is_dir() else a.resume
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["world_size"] != world or saved["config"]["data"] != c["data"] or saved["manifest_sha256"] != manifest_sha256:
            raise ValueError("Exact resume requires same world size and dataset configuration")
        adapter.load_state_dict(saved["adapter"])
        opt.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["scheduler"])
        step, epoch, offset = saved["step"], saved["epoch"], saved["offset"]
        state = saved["rng"][rank]
        rng.set_state(state["generator"])
        torch.set_rng_state(state["torch"])
        torch.cuda.set_rng_state(state["cuda"], device)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
    print(json.dumps({"rank": rank, "trainable": sum(p.numel() for p in adapter.parameters()),
                      "frozen": sum(p.numel() for p in pipeline.parameters())}), flush=True)
    while step < c["train"]["max_steps"]:
        sampler.set_epoch(epoch)
        timer = time.perf_counter()
        for batch_index, batch in enumerate(loader):
            if batch_index < offset:
                continue
            data_time = time.perf_counter() - timer
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            gt, history, real, cond, anchor_index, _ = prepare(pipeline, batch, device, c, rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = future_loss(pipeline, model, gt, history, real, cond, anchor_index, rng,
                                   c["reanchor"]["delta_regularization"])
            torch.cuda.synchronize()
            forward_end = time.perf_counter()
            loss.backward()
            grads = [p.grad for p in adapter.parameters()]
            finite = torch.tensor(int(torch.isfinite(loss).item() and all(g is not None and torch.isfinite(g).all() for g in grads)), device=device)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item() or any(p.grad is not None for p in pipeline.parameters()):
                raise RuntimeError("Nonfinite/missing adapter gradient or unfrozen backbone")
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), c["train"]["grad_clip"])
            if norm == 0:
                raise RuntimeError("Zero future-loss gradient: inspect conditioning path")
            torch.cuda.synchronize()
            backward_end = time.perf_counter()
            opt.step()
            schedule.step()
            torch.cuda.synchronize()
            end = time.perf_counter()
            step += 1
            offset = batch_index + 1
            print(json.dumps({"rank": rank, "step": step, "loss": loss.item(), "grad_norm": norm.item(),
                              "gate": adapter.gate_logit.sigmoid().item(), "data_time": data_time,
                              "forward_time": forward_end - start, "backward_time": backward_end - forward_end,
                              "optimizer_time": end - backward_end, "step_time": end - start + data_time,
                              "peak_vram": torch.cuda.max_memory_allocated()}), flush=True)
            if step % c["train"]["save_every"] == 0 or step == c["train"]["max_steps"]:
                local_rng = {"generator": rng.get_state(), "torch": torch.get_rng_state(),
                             "cuda": torch.cuda.get_rng_state(device), "python": random.getstate(), "numpy": np.random.get_state()}
                states = [None] * world
                if world > 1:
                    dist.all_gather_object(states, local_rng)
                else:
                    states[0] = local_rng
                if rank == 0:
                    folder = ROOT / f"checkpoints/step_{step:04d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    torch.save({"adapter": adapter.state_dict(), "optimizer": opt.state_dict(),
                                "scheduler": schedule.state_dict(), "step": step, "epoch": epoch,
                                "offset": offset, "world_size": world, "config": c, "rng": states,
                                "manifest_sha256": manifest_sha256}, folder / "state.tmp")
                    (folder / "state.tmp").replace(folder / "state.pt")
                    latest = ROOT / "checkpoints/latest.tmp"
                    latest.write_text(str(folder))
                    latest.replace(ROOT / "checkpoints/latest.txt")
            if step % c["train"]["eval_every"] == 0 or step == c["train"]["max_steps"]:
                # All ranks validate one fixed case to avoid DDP timeout waiting for rank 0.
                from eval_reanchor import evaluate
                evaluate(pipeline, adapter, c, device, ROOT / f"outputs/validation/step_{step}/rank_{rank}", cases=1)
            timer = time.perf_counter()
            if step >= c["train"]["max_steps"]:
                break
        epoch, offset = epoch + 1, 0
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
