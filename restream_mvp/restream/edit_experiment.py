"""Shared experiment plumbing for the Edit-Ready MVP scripts.

Keeps GPU entry points thin: config loading, prompt-case expansion, sharding,
deterministic noise sampling and provenance records (plan section 14).
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .edit_cache import (
    ROOT,
    canonical_hash,
    code_commit,
    config_hash,
    model_identity,
    sha256_file,
    sha256_text,
)
from .edit_replay import latent_shape

GIT_COMMIT = None


def read_config(path: os.PathLike | str) -> Dict[str, Any]:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def generation_geometry(config: Dict[str, Any]) -> Dict[str, Any]:
    shape = latent_shape(config)
    return {"latent_shape": list(shape), "num_latent_frames": shape[1],
            "height": int(config["generation"]["height"]),
            "width": int(config["generation"]["width"]),
            "fps": float(config["generation"]["fps"])}


def pixel_frame_count(config: Dict[str, Any]) -> int:
    frames = int(config["generation"]["num_latent_frames"])
    return 4 * (frames - 1) + 1


def sample_noise(config: Dict[str, Any], device: torch.device | str, seed: int) -> torch.Tensor:
    """Deterministic initial noise, mirroring ``inference.py`` sampling order.

    On CUDA this must stay ``torch.randn(..., dtype=torch.bfloat16)`` so the RNG
    stream matches upstream; the float32 fallback only ever triggers on a
    CPU-only host.
    """
    shape = latent_shape(config)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        return torch.randn(shape, device=device, dtype=torch.bfloat16)
    except RuntimeError:  # pragma: no cover - CPU-only fallback
        return torch.randn(shape, device=device, dtype=torch.float32).to(torch.bfloat16)


def prompt_cases(config: Dict[str, Any], limit: Optional[int] = None,
                 prompt_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    prompts = config["edit"]["prompts"]
    if prompt_ids:
        wanted = set(prompt_ids)
        prompts = [entry for entry in prompts if entry["id"] in wanted]
        missing = wanted - {entry["id"] for entry in prompts}
        if missing:
            raise ValueError(f"Unknown prompt ids: {sorted(missing)}")
    if limit is not None:
        prompts = prompts[:limit]
    return [dict(entry, index=index) for index, entry in enumerate(prompts)]


def target_chunks(config: Dict[str, Any], override: Optional[Sequence[int]] = None,
                  num_chunks: Optional[int] = None) -> List[int]:
    targets = list(override if override is not None else config["edit"]["targets"])
    if num_chunks is not None:
        invalid = [value for value in targets if not 0 <= value < num_chunks]
        if invalid:
            raise ValueError(f"Target chunks {invalid} outside [0, {num_chunks})")
    return [int(value) for value in targets]


def expand_cases(config: Dict[str, Any], cases: Optional[int] = None,
                 targets: Optional[Sequence[int]] = None,
                 seed_stride: int = 0,
                 prompt_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """One entry per (prompt case, target chunk)."""
    available = prompt_cases(config, limit=cases, prompt_ids=prompt_ids)
    chunk_targets = target_chunks(config, targets)
    expanded: List[Dict[str, Any]] = []
    for entry in available:
        for target in chunk_targets:
            expanded.append({
                "case_index": len(expanded),
                "prompt_id": entry["id"],
                "prompt_index": entry["index"],
                "base_prompt": entry["base"],
                "edit_prompt": entry["edit"],
                "probe": entry.get("probe"),
                "target_chunk": int(target),
                "seed": int(config["seed"]) + entry["index"] + seed_stride * len(available),
            })
    return expanded


def group_cases(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse (prompt, target) entries into one entry per prompt.

    The base video, the full-regeneration reference and the decoded base pixels
    are identical for every target chunk of the same prompt+seed, so the scripts
    compute them once per group.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        key = f"{case['prompt_id']}|{case['seed']}"
        group = groups.get(key)
        if group is None:
            group = {field: case[field] for field in
                     ("prompt_id", "prompt_index", "base_prompt", "edit_prompt", "probe", "seed")}
            group["targets"] = []
            groups[key] = group
        group["targets"].append(int(case["target_chunk"]))
    for group in groups.values():
        group["targets"] = sorted(set(group["targets"]))
    return list(groups.values())


def shard(items: Sequence[Any], index: int, count: int) -> List[Any]:
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"Invalid shard {index}/{count}")
    return [item for position, item in enumerate(items) if position % count == index]


def write_json(path: os.PathLike | str, payload: Any) -> None:
    """Write strict JSON, replacing non-finite floats with ``null``.

    ``psnr`` is ``inf`` for a bit-identical comparison, which is not valid JSON.
    Every replacement is listed under ``non_finite_fields`` so nothing is hidden.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    replaced: List[str] = []
    cleaned = _json_safe(payload, replaced)
    if replaced and isinstance(cleaned, dict):
        cleaned.setdefault("non_finite_fields", replaced)
    text = json.dumps(cleaned, indent=2, allow_nan=False, sort_keys=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text + "\n")
    os.replace(temporary, path)


def _json_safe(value: Any, replaced: List[str], path: str = "") -> Any:
    import math

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        replaced.append(path or "<root>")
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item, replaced, f"{path}.{key}" if path else str(key))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, replaced, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.tolist(), replaced, path)
    return value


def run_provenance(config: Dict[str, Any], config_path: os.PathLike | str, *,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global GIT_COMMIT
    if GIT_COMMIT is None:
        GIT_COMMIT = code_commit()
    record = {
        "git_commit": GIT_COMMIT,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_hash": config_hash(config),
        "model_identity": model_identity(config),
        "cache_schema": 1,
        "host": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "created_unix": time.time(),
    }
    record["model_checkpoint_sha256"] = record["model_identity"]["model_hash"]
    if extra:
        record.update(extra)
    return record


def peak_vram_bytes(device: torch.device | str) -> int:
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated(device))


def device_for(argument: Optional[int] = None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if argument is None:
        index = 0
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
        return torch.device(f"cuda:{index}")
    return torch.device(f"cuda:{int(argument)}")


def prompt_hash(text: str) -> str:
    return sha256_text(text)
