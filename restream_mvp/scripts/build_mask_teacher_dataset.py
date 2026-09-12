#!/usr/bin/env python3
"""Merge existing and new Oracle masks into the 40-unit M1-A teacher dataset."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream.mask_distillation import PROTOCOL, checkpoint_state_feature, feature_digest  # noqa: E402
from restream.mask_distillation_protocol import EDITS, TEACHER_SEEDS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-shards", type=Path, nargs=4, required=True)
    parser.add_argument("--existing-results", type=Path, nargs=4, required=True)
    parser.add_argument("--existing-cache-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs/mask_distillation_plan.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    expected_existing_hashes = set(plan["teacher"]["existing_result_sha256"])
    if {ec.sha256_file(path) for path in args.existing_results} != expected_existing_hashes:
        raise ValueError("existing Oracle result hashes differ from frozen plan")
    records, prompt_by_edit, sources = {}, {}, []
    identities = set()
    for path in args.new_shards:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("experiment") != "mask_distillation_teacher_m1a":
            raise ValueError(f"incomplete new teacher shard {path}")
        provenance = payload["provenance"]
        if provenance["git_dirty"]:
            raise ValueError("new teacher provenance is dirty")
        identities.add((provenance["model_checkpoint_sha256"], provenance["config_hash"]))
        sources.append({"path": str(path), "sha256": ec.sha256_file(path)})
        for item in payload["teachers"]:
            key = (item["prompt_id"], item["seed"])
            if key in records:
                raise ValueError(f"duplicate teacher {key}")
            prompt = torch.tensor(item["prompt_feature"], dtype=torch.float32)
            state = torch.tensor(item["state_feature"], dtype=torch.float32)
            if feature_digest(prompt, state) != item["feature_sha256"]:
                raise ValueError(f"feature digest mismatch {key}")
            previous = prompt_by_edit.setdefault(item["prompt_id"], prompt)
            if not torch.equal(previous, prompt):
                raise ValueError(f"prompt feature changed across seeds for {item['prompt_id']}")
            records[key] = {"prompt_id": item["prompt_id"], "seed": item["seed"],
                            "target_chunk": 4, "mask": item["mask"],
                            "prompt_feature": prompt, "state_feature": state,
                            "feature_sha256": item["feature_sha256"],
                            "teacher_metrics": {"editability": item["editability"],
                                                "D_drift": item["D_drift"],
                                                "best_loss": item["best_loss"]},
                            "source": "new_oracle"}
    for path in args.existing_results:
        payload = json.loads(path.read_text())
        identities.add((payload["provenance"]["model_checkpoint_sha256"],
                        payload["provenance"]["config_hash"]))
        sources.append({"path": str(path), "sha256": ec.sha256_file(path)})
        for item in payload["cases"]:
            if item["condition"] != "oracle_lambda_0.2":
                continue
            key = (item["prompt_id"], item["seed"])
            matches = list(args.existing_cache_root.glob(
                f"shard*_cache/oracle_{item['prompt_id']}_seed{item['seed']}__chunk004.pt"))
            if len(matches) != 1:
                raise ValueError(f"expected one existing checkpoint for {key}, got {matches}")
            checkpoint = ec.load_edit_checkpoint(matches[0])
            state = checkpoint_state_feature(checkpoint)
            prompt = prompt_by_edit[item["prompt_id"]]
            records[key] = {"prompt_id": item["prompt_id"], "seed": item["seed"],
                            "target_chunk": 4, "mask": item["layer_release"],
                            "prompt_feature": prompt, "state_feature": state,
                            "feature_sha256": feature_digest(prompt, state),
                            "teacher_metrics": {"editability": item["editability"],
                                                "D_drift": item["D_drift"],
                                                "best_loss": item["optimization"]["best_loss"]},
                            "source": "existing_oracle_312580a"}
    expected = {(edit, seed) for edit in EDITS for seed in TEACHER_SEEDS}
    if set(records) != expected or len(identities) != 1:
        raise ValueError("teacher grid incomplete or model/config identity mismatch")
    payload = {"schema": 1, "protocol": PROTOCOL, "status": "complete",
               "plan_sha256": ec.sha256_file(args.plan), "model_config_identity": list(identities)[0],
               "sources": sources, "teachers": [records[key] for key in sorted(records)]}
    ex.write_json(args.output, payload)
    print(f"Teacher dataset: {args.output} ({len(records)} units)")


if __name__ == "__main__":
    main()
