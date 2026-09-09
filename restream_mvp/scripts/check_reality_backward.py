"""One real R0 teacher-forcing backward, no optimizer and no training checkpoint."""
import argparse
import copy
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.objective import future_loss
from restream.reality_data import write_json
from restream.reality_dataset import collate_reality
from restream.reality_runtime import (read_reality_config, make_memory, make_dataset,
                                      prepare_reality, reality_loss, PreserveHistory)
from restream.runtime import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/real_backward.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Real backward check requires CUDA")
    config = read_reality_config(args.config)
    torch.manual_seed(config["seed"])
    device = torch.device("cuda")
    dataset = make_dataset(config, "train")
    index = next(i for i, row in enumerate(dataset.rows) if row["reference_kind"] == "async")
    batch = collate_reality([dataset[index]])
    print("Loading frozen LongLive for R0 graph validation", flush=True)
    pipeline = load_pipeline(config, device)
    memory = make_memory(config).to(device)
    gt, conditioning, anchor = prepare_reality(pipeline, batch, device, config)
    features, mask = batch["features"].to(device), batch["reference_mask"].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        initial, _ = memory(conditioning["prompt_embeds"], features, mask)
        absent, _ = memory(conditioning["prompt_embeds"], features, torch.zeros_like(mask))
        assert torch.equal(initial, conditioning["prompt_embeds"])
        assert torch.equal(absent, conditioning["prompt_embeds"])
        base = future_loss(pipeline, PreserveHistory(), gt, gt[:, :anchor + 1], gt[:, anchor:anchor + 1],
                           conditioning, anchor, torch.Generator(device=device).manual_seed(123), 0)
    check_config = copy.deepcopy(config)
    check_config["reality_memory"]["regularization"] = {"delta_weight": 0, "wrong_gate_weight": 0}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, stats = reality_loss(pipeline, memory, gt, conditioning, anchor, features, mask,
                                   batch["wrong_reference"].to(device), torch.Generator(device=device).manual_seed(123),
                                   check_config, training=False)
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, base, rtol=0, atol=0)
    print(f"Real future loss={loss.item():.6f}; running backward", flush=True)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
    norm = sum(p.grad.float().square().sum() for p in memory.parameters()).sqrt().item()
    assert norm > 0
    assert all(not p.requires_grad and p.grad is None for p in pipeline.parameters())
    report = {"status": "passed", "config": config, "sample_id": batch["sample_id"][0],
              "optimizer_steps": 0, "regularization_weights": [0, 0], "loss": loss.item(), "base_loss": base.item(),
              "initial_context_exact_base": True, "no_memory_context_exact_base": True,
              "memory_grad_norm": norm, "trainable_parameters": sum(p.numel() for p in memory.parameters()),
              "parameter_grad_norms": {name: p.grad.float().norm().item() for name, p in memory.named_parameters()},
              "memory_gate": stats["gate"].mean().item(), "memory_attention_entropy": stats["attention_entropy"].mean().item(),
              "backbone_gradients_none": True, "peak_vram_bytes": torch.cuda.max_memory_allocated(),
              "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
