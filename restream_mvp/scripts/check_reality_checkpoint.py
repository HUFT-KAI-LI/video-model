"""Audit saved NCCL smoke checkpoints and resumed rank logs, without another model run."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_data import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--before", type=int, default=3)
    parser.add_argument("--after", type=int, default=4)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/overfit_review/nccl_resume.json")
    args = parser.parse_args()
    before_path, after_path = [args.run / f"step_{step:04d}/state.pt" for step in (args.before, args.after)]
    before, after = [torch.load(path, map_location="cpu", weights_only=False) for path in (before_path, after_path)]
    if before["signature"] != after["signature"] or after["signature"]["world_size"] != args.world_size:
        raise ValueError("Checkpoint signature/world size differs")
    if before["step"] != args.before or after["step"] != args.after:
        raise ValueError("Unexpected saved step")
    if len(before["rng"]) != args.world_size or len(after["rng"]) != args.world_size:
        raise ValueError("Missing per-rank RNG state")
    if after["optimizer_steps"] <= before["optimizer_steps"]:
        raise ValueError("Resumed run performed no optimizer update")
    logs = [[json.loads(line) for line in (args.run / f"train_rank_{rank}.jsonl").read_text().splitlines()]
            for rank in range(args.world_size)]
    for rows in logs:
        if [row["step"] for row in rows] != list(range(1, args.after + 1)):
            raise ValueError("Missing/repeated rank training steps")
    for step in range(args.after):
        reference = logs[0][step]
        for rank in range(1, args.world_size):
            if logs[rank][step]["gradient_norms"] != reference["gradient_norms"]:
                raise ValueError("Post-allreduce gradient norms disagree between ranks")
    changed = [name for name in before["memory"] if not torch.equal(before["memory"][name], after["memory"][name])]
    if not changed or any(not torch.isfinite(tensor).all() for tensor in after["memory"].values()):
        raise ValueError("Resumed weights unchanged/nonfinite")
    result = {"status": "passed", "world_size": args.world_size, "backend": "nccl",
              "saved_step": args.before, "resumed_step": args.after,
              "optimizer_steps_before": before["optimizer_steps"], "optimizer_steps_after": after["optimizer_steps"],
              "all_rank_gradient_norms_identical": True, "changed_parameter_tensors_after_resume": changed,
              "rng_states": len(after["rng"]),
              "checkpoint_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (before_path, after_path)},
              "note": "Confirms real save/restart/resume and rank synchronization; not a bitwise comparison against uninterrupted training."}
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
