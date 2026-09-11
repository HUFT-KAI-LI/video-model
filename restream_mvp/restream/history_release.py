"""Validated case selection and sealed references for fixed-history interventions."""
import json
import math
from pathlib import Path

from . import edit_experiment as ex
from . import edit_cache as ec

PROTOCOL = "fixed_history_paired_v2"


def manifest_groups(config, manifest):
    if manifest.get("schema") != 2:
        raise ValueError("Use schema 2: one case per edit/seed/target/gate; P0/P1 run internally")
    entries = manifest.get("cases", [])
    if not entries:
        raise ValueError("manifest has no cases")
    prompts = {p["id"]: p for p in ex.prompt_cases(config)}
    groups, seen = {}, set()
    for entry in entries:
        if "policy" in entry:
            raise ValueError("policy must not be a manifest dimension; every case executes P0 and P1")
        prompt = prompts.get(entry["edit"])
        if prompt is None:
            raise ValueError(f"Unknown edit {entry['edit']}")
        seed, target = entry["seed"], entry["target_chunk"]
        if type(seed) is not int or type(target) is not int or target < 0:
            raise ValueError("seed and target_chunk must be integers; target_chunk must be nonnegative")
        gate = float(entry["history_gate"])
        if not math.isfinite(gate) or not 0 <= gate <= 1:
            raise ValueError("history_gate must be finite and in [0,1]")
        key = (entry["edit"], seed, target, gate)
        if key in seen:
            raise ValueError(f"Duplicate manifest pair {key}")
        seen.add(key)
        group_key = (entry["edit"], seed)
        if group_key not in groups:
            groups[group_key] = {"prompt_id": prompt["id"], "prompt_index": prompt["index"],
                                 "base_prompt": prompt["base"], "edit_prompt": prompt["edit"],
                                 "probe": prompt.get("probe"), "evidence": prompt.get("evidence", "directional"),
                                 "seed": seed, "targets": [], "gates_by_target": {}}
        group = groups[group_key]
        group["gates_by_target"].setdefault(target, []).append(gate)
    for group in groups.values():
        group["targets"] = sorted(group["gates_by_target"])
    return list(groups.values())


def validate_smoke(groups):
    if len(groups) != 1 or groups[0]["targets"] != [0, 1, 4]:
        raise ValueError("invariant smoke requires 1 edit x 1 seed x chunks {0,1,4}")
    if any(set(values) != {1.0, 0.5, 0.0} for values in groups[0]["gates_by_target"].values()):
        raise ValueError("invariant smoke requires gates {1,.5,0} at every target")


def load_sealed(paths, groups, identity):
    """Fail before generation if any g=1 reference is absent or incompatible."""
    wanted = {(g["prompt_id"], g["seed"], k): g for g in groups for k in g["targets"]
              if 1.0 in g["gates_by_target"][k]}
    records, sources = {}, []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        sources.append({"path": str(path), "sha256": ec.sha256_file(path),
                        "git_commit": payload.get("provenance", {}).get("git_commit")})
        for record in payload.get("cases", []):
            key = (record["prompt_id"], record["seed"], record["target_chunk"])
            if key not in wanted:
                continue
            if record.get("history_gate", 1.0) != 1.0:
                continue
            if key in records:
                raise ValueError(f"Ambiguous sealed case {key}")
            for field in ("config_hash", "model_checkpoint_sha256"):
                if payload["provenance"][field] != identity[field]:
                    raise ValueError(f"Sealed {field} mismatch for {key}")
            for field in ("base_prompt", "edit_prompt"):
                if record[field] != wanted[key][field]:
                    raise ValueError(f"Sealed {field} mismatch for {key}")
            for variant in ("original", "text_rebind", "full_regeneration"):
                if not record.get("chunk_latent_sha256", {}).get(variant):
                    raise ValueError(f"Sealed {variant} digest missing for {key}")
            records[key] = record
    if wanted.keys() - records.keys():
        raise ValueError(f"Missing sealed cases: {sorted(wanted.keys() - records.keys())}")
    return records, sources
