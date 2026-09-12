#!/usr/bin/env python3
"""Merge the locked 40-unit M1-A dataset with 160 new M1-B teachers."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from restream import edit_cache as ec  # noqa: E402
from restream import edit_experiment as ex  # noqa: E402
from restream.mask_distillation import feature_digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--new-shards", type=Path, nargs=4, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    teacher_plan = plan["teacher"]
    if ec.sha256_file(args.base_dataset) != teacher_plan["base_dataset_sha256"]:
        raise ValueError("M1-A base teacher dataset hash mismatch")
    base = json.loads(args.base_dataset.read_text())
    if base.get("status") != "complete" or len(base.get("teachers", [])) != 40:
        raise ValueError("base teacher dataset must contain the locked 40 M1-A units")

    records, prompt_by_edit, sources = {}, {}, [{
        "path": str(args.base_dataset), "sha256": ec.sha256_file(args.base_dataset),
        "units": len(base["teachers"]), "role": "locked_m1a_base"}]
    identity = tuple(base["model_config_identity"])

    def add(item, source):
        key = (item["prompt_id"], int(item["seed"]))
        if key in records:
            raise ValueError(f"duplicate teacher {key}")
        prompt = torch.tensor(item["prompt_feature"], dtype=torch.float32)
        state = torch.tensor(item["state_feature"], dtype=torch.float32)
        if feature_digest(prompt, state) != item["feature_sha256"]:
            raise ValueError(f"feature digest mismatch {key}")
        previous = prompt_by_edit.setdefault(item["prompt_id"], prompt)
        if not torch.equal(previous, prompt):
            raise ValueError(f"prompt feature changed across seeds for {item['prompt_id']}")
        records[key] = {"prompt_id": item["prompt_id"], "seed": int(item["seed"]),
                        "target_chunk": 4, "mask": item["mask"],
                        "prompt_feature": item["prompt_feature"],
                        "state_feature": item["state_feature"],
                        "feature_sha256": item["feature_sha256"],
                        "teacher_metrics": item["teacher_metrics"], "source": source}

    for item in base["teachers"]:
        add(item, "locked_m1a_teacher")
    for path in args.new_shards:
        payload = json.loads(path.read_text())
        if (payload.get("status") != "complete"
                or payload.get("experiment") != teacher_plan["experiment"]):
            raise ValueError(f"incomplete M1-B teacher shard {path}")
        provenance = payload["provenance"]
        if provenance["git_dirty"] or (provenance["model_checkpoint_sha256"], provenance["config_hash"]) != identity:
            raise ValueError("dirty or mismatched M1-B teacher provenance")
        sources.append({"path": str(path), "sha256": ec.sha256_file(path),
                        "units": len(payload["teachers"]), "role": "new_m1b_teacher"})
        for item in payload["teachers"]:
            add({**item, "teacher_metrics": {"editability": item["editability"],
                                              "D_drift": item["D_drift"],
                                              "best_loss": item["best_loss"]}},
                "new_m1b_oracle")

    expected = {(edit, seed) for edit in teacher_plan["edits"]
                for seed in teacher_plan["all_seeds"]}
    if set(records) != expected or len(records) != teacher_plan["expected_total_units"]:
        raise ValueError("M1-B teacher grid is incomplete")
    payload = {"schema": 1, "protocol": plan["protocol"], "status": "complete",
               "plan_sha256": ec.sha256_file(args.plan),
               "model_config_identity": list(identity), "sources": sources,
               "teachers": [records[key] for key in sorted(records)]}
    ex.write_json(args.output, payload)
    print(f"Teacher dataset: {args.output} ({len(records)} units)")


if __name__ == "__main__":
    main()
