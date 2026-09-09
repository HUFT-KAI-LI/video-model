"""Two real AdamW updates verify gradients beyond the zero-initialized output layer."""
import argparse
import copy
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json
from restream.reality_dataset import collate_reality
from restream.reality_diagnostics import gradient_norms
from restream.reality_runtime import read_reality_config, make_memory, make_dataset, prepare_reality, reality_loss
from restream.runtime import load_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--reviewed", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/overfit_review/optimizer_sanity.json")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("This check performs two real optimizer updates; pass --reviewed after review")
    config = read_reality_config(args.config)
    torch.manual_seed(config["seed"])
    dataset = make_dataset(config, "train")
    index = next(i for i, row in enumerate(dataset.rows) if row["reference_kind"] == "async")
    batch = collate_reality([dataset[index]])
    device = torch.device("cuda")
    print("Loading real frozen LongLive for two-step optimizer sanity", flush=True)
    pipeline = load_pipeline(config, device)
    memory = make_memory(config).to(device)
    optimizer = torch.optim.AdamW(memory.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"])
    gt, cond, anchor = prepare_reality(pipeline, batch, device, config)
    check_config = copy.deepcopy(config)
    check_config["reality_memory"]["regularization"] = {"delta_weight": 0, "wrong_gate_weight": 0}
    records = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, stats = reality_loss(pipeline, memory, gt, cond, anchor, batch["features"].to(device),
                                       batch["reference_mask"].to(device), batch["wrong_reference"].to(device),
                                       torch.Generator(device=device).manual_seed(123), check_config, training=False)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite optimizer sanity loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters()):
            raise RuntimeError("Missing/nonfinite memory gradient")
        norms = gradient_norms(memory)
        if norms["output"] <= 0 or (step == 2 and any(value is None or value <= 0 for value in norms.values())):
            raise RuntimeError(f"Step {step} did not activate all memory gradient groups: {norms}")
        if any(p.grad is not None or p.requires_grad for p in pipeline.parameters()):
            raise RuntimeError("Backbone was not kept frozen")
        torch.nn.utils.clip_grad_norm_(memory.parameters(), config["train"]["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            absent, _ = memory(cond["prompt_embeds"], batch["features"].to(device), torch.zeros_like(batch["reference_mask"], device=device))
            if not torch.equal(absent, cond["prompt_embeds"]):
                raise RuntimeError("Optimizer update broke no-memory invariant")
        record = {"step": step, "video_loss": loss.item(), "gradient_norms": norms,
                  "gate": stats["gate"].mean().item(), "no_memory_exact_base": True}
        records.append(record)
        print(record, flush=True)
    write_json(args.output, {"status": "passed", "config": config, "sample_id": batch["sample_id"][0],
                            "steps": records, "optimizer_steps": 2, "regularization_weights": [0, 0],
                            "dropout": False, "noise_seed": 123, "backbone_gradients_none": True,
                            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                            "note": "Separate sanity model; weights discarded before the fresh 10-step run."})


if __name__ == "__main__":
    main()
