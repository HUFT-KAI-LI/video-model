#!/usr/bin/env python3
"""Train prompt-only and prompt+state M1-A controllers on Oracle masks."""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream.mask_distillation import PromptOnlyController, PromptStateController, normalize  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mask_distillation_plan.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    plan, dataset = json.loads(args.plan.read_text()), json.loads(args.dataset.read_text())
    protocol = plan["protocol"]
    expected_units = int(plan["teacher"]["expected_total_units"])
    if (dataset.get("protocol") != protocol or dataset.get("status") != "complete"
            or len(dataset["teachers"]) != expected_units):
        raise ValueError(f"controller training requires complete {expected_units}-unit teacher dataset")
    cfg, seed = plan["controllers"], int(plan["controllers"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    prompt = torch.tensor([row["prompt_feature"] for row in dataset["teachers"]], dtype=torch.float32)
    state = torch.tensor([row["state_feature"] for row in dataset["teachers"]], dtype=torch.float32)
    target = torch.tensor([row["mask"] for row in dataset["teachers"]], dtype=torch.float32)
    prompt_mean, prompt_std = prompt.mean(0), prompt.std(0, unbiased=False).clamp_min(1e-6)
    state_mean, state_std = state.mean((0, 1)), state.std((0, 1), unbiased=False).clamp_min(1e-6)
    prompt_n, state_n = normalize(prompt, prompt_mean, prompt_std), normalize(state, state_mean, state_std)
    models = {"prompt_only": PromptOnlyController(), "prompt_state": PromptStateController()}
    histories = {}
    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
        history = []
        for epoch in range(int(cfg["epochs"])):
            prediction = model(prompt_n) if name == "prompt_only" else model(prompt_n, state_n)
            loss = torch.nn.functional.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if epoch in (0, 9, 99, 499, 999, int(cfg["epochs"]) - 1):
                history.append({"epoch": epoch + 1, "mask_mse": float(loss.detach())})
        histories[name] = history
    with torch.no_grad():
        predictions = {"prompt_only": models["prompt_only"](prompt_n),
                       "prompt_state": models["prompt_state"](prompt_n, state_n)}
    normalization = {"prompt_mean": prompt_mean, "prompt_std": prompt_std,
                     "state_mean": state_mean, "state_std": state_std}
    checkpoint = {"schema": 1, "protocol": protocol,
                  "prompt_only_state_dict": models["prompt_only"].state_dict(),
                  "prompt_state_state_dict": models["prompt_state"].state_dict(),
                  "normalization": normalization,
                  "metadata": {"dataset_sha256": ec.sha256_file(args.dataset),
                               "plan_sha256": ec.sha256_file(args.plan), "seed": seed,
                               "epochs": cfg["epochs"], "learning_rate": cfg["learning_rate"]}}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(".tmp")
    torch.save(checkpoint, temporary); temporary.replace(args.checkpoint)
    report = {"schema": 1, "protocol": protocol, "status": "complete",
              "dataset_sha256": ec.sha256_file(args.dataset),
              "controller_checkpoint_sha256": ec.sha256_file(args.checkpoint),
              "histories": histories,
              "final_mask_mse": {name: float(torch.nn.functional.mse_loss(value, target))
                                 for name, value in predictions.items()},
              "parameter_counts": {name: sum(p.numel() for p in model.parameters())
                                   for name, model in models.items()},
              "note": "Training-set mask MSE is diagnostic only; held-out replay decides the experiment."}
    ex.write_json(args.report, report)
    print(json.dumps(report["final_mask_mse"], indent=2))


if __name__ == "__main__":
    main()
