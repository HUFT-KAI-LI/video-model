"""Run one real LongLive teacher-forcing backward pass; never updates weights."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from restream.anchor_adapter import GatedAnchorAdapter
from restream.dataset import VideoDataset
from restream.objective import future_loss, prepare
from restream.runtime import load_pipeline, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/restream_mvp.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/real_backward_check.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real teacher-forcing check")
    config = read_config(args.config)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda", torch.cuda.current_device())
    print("Loading frozen Wan + LongLive + official LoRA", flush=True)
    pipeline = load_pipeline(config, device)
    dataset = VideoDataset(ROOT / config["data"]["train_manifest"], config["data"]["frames"],
                           config["data"]["height"], config["data"]["width"], config["data"]["fps"])
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    adapter = GatedAnchorAdapter(config["reanchor"]["channels"], config["reanchor"]["gate_init_logit"]).to(device)
    rng = torch.Generator(device=device).manual_seed(config["seed"])
    print("Encoding real training sample", flush=True)
    gt, history, real, conditioning, index, actual_time = prepare(pipeline, batch, device, config, rng, force_drift=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if not torch.equal(adapter(history[:, -1:], real), history[:, -1:]):
            raise RuntimeError("Step-zero adapter must preserve its input exactly")
    print("Running real teacher-forcing forward", flush=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = future_loss(pipeline, adapter, gt, history, real, conditioning, index, rng,
                           0)  # Future supervision alone must produce the gradient.
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite real loss: {loss.item()}")
    print(f"Backward, loss={loss.item():.6f}", flush=True)
    loss.backward()
    adapter_grads = [p.grad for p in adapter.parameters()]
    if not all(g is not None and torch.isfinite(g).all() for g in adapter_grads):
        raise RuntimeError("Adapter gradient is missing or non-finite")
    if any(p.grad is not None for p in pipeline.parameters()):
        raise RuntimeError("Frozen LongLive backbone received a gradient")
    grad_norm = torch.sqrt(sum(g.float().square().sum() for g in adapter_grads)).item()
    if not grad_norm > 0:
        raise RuntimeError("Future supervision produced zero adapter gradient")
    result = {
        "status": "passed",
        "config": config,
        "source_id": batch["source_id"][0],
        "seed": config["seed"],
        "optimizer_steps": 0,
        "delta_regularization": 0,
        "adapter_identity_at_init": True,
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "loss": float(loss.detach().cpu()),
        "anchor_latent_index": int(index),
        "actual_anchor_sec": float(actual_time),
        "adapter_grad_norm": grad_norm,
        "parameter_grad_norms": {name: p.grad.float().norm().item() for name, p in adapter.named_parameters()},
        "backbone_grads_clear": True,
        "backbone_frozen": all(not p.requires_grad for p in pipeline.parameters()),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
