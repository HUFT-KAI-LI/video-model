"""Generation-Time Edit Cache: capture, persist and restore LongLive AR chunk state.

Feasibility MVP for `EDIT_READY_VIDEO_FEASIBILITY_PLAN.md`.

Scope (this round): pure inference / state reuse.  No training, no adapter, no
spatial mask, no bidirectional propagation.  This module only answers *what has to
be saved at a chunk boundary so that a single already-generated chunk can be
reopened later*.

Backbone facts this schema is built on (see ``EDIT_READY_MVP.md`` for the loop
diagram and the upstream file/line evidence):

* ``CausalInferencePipeline.inference`` walks ``num_frame_per_block`` latent frames
  per AR step.  Between two AR steps the only mutable runtime state is
  ``pipeline.kv_cache1`` (30 blocks x {k, v, global_end_index, local_end_index}),
  ``pipeline.crossattn_cache`` (30 blocks x {k, v, is_init}) and
  ``current_start_frame``.
* Sink tokens are *not* a separate object: they live in the first
  ``sink_size * frame_seq_length`` slots of ``kv_cache1``.  ``sink_cache`` is kept
  in the schema for recall but is ``None`` for this backbone.
* The flow-match scheduler and the RoPE frequency table are derived from config,
  they carry no per-chunk mutable state; we still record their digest/shape as
  provenance.
* Chunk ``k`` consumes one saved initial noise slice plus
  ``len(denoising_step_list) - 1`` in-chunk noise draws (``torch.randn_like`` on
  the default CUDA generator).  Both the draws and the RNG states that produce
  them are recorded so replay can be verified bit-for-bit.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
PAYLOAD_TYPE = "restream.edit_checkpoint"


# --------------------------------------------------------------------------- #
# hashing helpers
# --------------------------------------------------------------------------- #
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: os.PathLike | str, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def config_hash(config: Any) -> str:
    """Stable digest of a config mapping (dict / OmegaConf / dataclass)."""
    if hasattr(config, "items"):
        value = {str(k): config[k] for k in config}
    else:
        value = config
    return canonical_hash(_jsonable(value))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def code_commit() -> str:
    """Best-effort git commit of the repository containing this file."""
    import subprocess

    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        commit = out.stdout.strip()
        if out.returncode == 0 and commit:
            dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=10).stdout.strip()
            return commit + ("+dirty" if dirty else "")
    except Exception:  # pragma: no cover - git is optional at runtime
        pass
    return "unknown"


def model_identity(config: Any) -> Dict[str, Any]:
    """Checkpoint identity without re-hashing multi-GB weights.

    LongLive base/LoRA digests come from the verified official record; the Wan
    files are identified by recorded byte sizes plus a head/tail digest.  The
    combined ``model_hash`` is the canonical hash of that record.
    """
    longlive = ROOT / config["model"]["longlive"]
    wan = ROOT / config["model"]["wan"]
    record: Dict[str, Any] = {"longlive": {}, "wan": {}}

    verified = longlive / "official_sha256_verified.json"
    if verified.is_file():
        record["longlive"] = json.loads(verified.read_text())
    else:
        for name in ("models/longlive_base.pt", "models/lora.pt"):
            candidate = longlive / name
            record["longlive"][name] = f"size:{candidate.stat().st_size}" if candidate.is_file() else "missing"

    for name in ("Wan2.1_VAE.pth", "diffusion_pytorch_model.safetensors",
                 "models_t5_umt5-xxl-enc-bf16.pth"):
        candidate = wan / name
        if not candidate.is_file():
            record["wan"][name] = "missing"
            continue
        size = candidate.stat().st_size
        record["wan"][name] = {"bytes": size, "edge_sha256": _edge_digest(candidate)}
    record["model_hash"] = canonical_hash(record)
    return record


def _edge_digest(path: Path, window: int = 1 << 20) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        digest.update(stream.read(window))
        if size > 2 * window:
            stream.seek(size - window)
            digest.update(stream.read(window))
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# tensor provenance / device movement
# --------------------------------------------------------------------------- #
def tensor_provenance(value: Any, prefix: str = "") -> Dict[str, Dict[str, Any]]:
    """Recursively record shape/dtype/device for every tensor in a structure."""
    recorded: Dict[str, Dict[str, Any]] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, torch.Tensor):
            recorded[path] = {"shape": list(node.shape), "dtype": str(node.dtype),
                              "device": str(node.device), "numel": int(node.numel())}
        elif isinstance(node, dict):
            for key, item in node.items():
                walk(item, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(value, prefix)
    return recorded


def structure_bytes(value: Any) -> int:
    total = 0
    if isinstance(value, torch.Tensor):
        total += value.numel() * value.element_size()
    elif isinstance(value, dict):
        for item in value.values():
            total += structure_bytes(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            total += structure_bytes(item)
    return total


def to_cpu(value: Any) -> Any:
    """Deep-copy a nested structure onto CPU, detaching tensors from the graph."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous().clone()
    if isinstance(value, dict):
        return {key: to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(item) for item in value)
    return value


def to_device(value: Any, device: torch.device | str) -> Any:
    """Move every tensor in a structure to ``device`` preserving dtype."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


# --------------------------------------------------------------------------- #
# checkpoint schema
# --------------------------------------------------------------------------- #
@dataclass
class EditCheckpoint:
    """Minimal recoverable state at the start of AR chunk ``chunk_index``."""

    # ---- identity / schema -------------------------------------------------
    schema_version: int
    sample_id: str
    chunk_index: int
    num_chunks: int

    # ---- provenance --------------------------------------------------------
    model_hash: str
    model_identity: Dict[str, Any]
    config_hash: str
    prompt: str
    prompt_hash: str
    seed: int
    code_commit: str
    created_unix: float

    # ---- generation geometry ----------------------------------------------
    num_frame_per_block: int
    frame_seq_length: int
    local_attn_size: int
    sink_size: int
    latent_shape: List[int]
    latent_dtype: str
    source_device: str
    current_start_frame: int

    # ---- reproducibility state --------------------------------------------
    torch_rng_state: torch.Tensor
    cuda_rng_state: List[torch.Tensor]
    generator_state: Optional[torch.Tensor]

    # ---- streaming generation state ---------------------------------------
    latent_history: Optional[torch.Tensor]
    kv_cache: Optional[List[Dict[str, Any]]]
    crossattn_cache: Optional[List[Dict[str, Any]]]
    sink_cache: Optional[List[Dict[str, Any]]]
    position_state: Optional[Dict[str, Any]]
    scheduler_state: Optional[Dict[str, Any]]
    conditional_dict: Optional[Dict[str, Any]]

    # ---- optional debugging ------------------------------------------------
    previous_chunk_latent: Optional[torch.Tensor] = None
    next_noise: Optional[torch.Tensor] = None
    denoise_noise: Optional[List[torch.Tensor]] = None

    provenance: Dict[str, Any] = field(default_factory=dict)

    # -- payload conversion --------------------------------------------------
    def to_payload(self) -> Dict[str, Any]:
        payload = {"payload_type": PAYLOAD_TYPE, "schema_version": self.schema_version}
        for name, value in self.__dict__.items():
            if name == "schema_version":
                continue
            payload[name] = to_cpu(value)
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "EditCheckpoint":
        if not isinstance(payload, dict) or payload.get("payload_type") != PAYLOAD_TYPE:
            raise ValueError("Not a ReStream edit checkpoint payload")
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"Unsupported edit checkpoint schema {payload.get('schema_version')!r}")
        body = {key: value for key, value in payload.items()
                if key not in ("payload_type", "schema_version")}
        body.setdefault("previous_chunk_latent", None)
        body.setdefault("next_noise", None)
        body.setdefault("denoise_noise", None)
        body.setdefault("provenance", {})
        return cls(schema_version=SCHEMA_VERSION, **body)

    # -- convenience ---------------------------------------------------------
    def cache_bytes(self) -> int:
        return structure_bytes(self.kv_cache) + structure_bytes(self.crossattn_cache)

    def identity(self) -> Dict[str, Any]:
        return {"schema_version": self.schema_version, "sample_id": self.sample_id,
                "chunk_index": self.chunk_index, "num_chunks": self.num_chunks,
                "prompt_hash": self.prompt_hash, "model_hash": self.model_hash,
                "config_hash": self.config_hash, "seed": self.seed,
                "current_start_frame": self.current_start_frame,
                "latent_shape": self.latent_shape}


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #
def save_edit_checkpoint(path: os.PathLike | str, checkpoint: EditCheckpoint,
                         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Atomically persist a checkpoint; return its manifest entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint.to_payload()
    if extra:
        payload.setdefault("provenance", {})
        payload["provenance"] = dict(payload["provenance"], **extra)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "chunk_index": checkpoint.chunk_index, "sample_id": checkpoint.sample_id,
            "cache_bytes": checkpoint.cache_bytes(), "identity": checkpoint.identity(),
            "created_unix": checkpoint.created_unix}


def load_edit_checkpoint(path: os.PathLike | str, device: torch.device | str | None = None,
                         verify_sha256: Optional[str] = None) -> EditCheckpoint:
    path = Path(path)
    if verify_sha256 is not None and sha256_file(path) != verify_sha256:
        raise ValueError(f"Edit checkpoint checksum mismatch: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # ``weights_only`` refuses nothing we store, but keep a documented fallback
        # so an older payload written by this same module still loads.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = EditCheckpoint.from_payload(payload)
    return checkpoint if device is None else checkpoint_to_device(checkpoint, device)


def checkpoint_to_device(checkpoint: EditCheckpoint, device: torch.device | str) -> EditCheckpoint:
    """Return a copy of the checkpoint whose GPU-bound tensors live on ``device``."""
    checkpoint.kv_cache = to_device(checkpoint.kv_cache, device)
    checkpoint.crossattn_cache = to_device(checkpoint.crossattn_cache, device)
    checkpoint.sink_cache = to_device(checkpoint.sink_cache, device)
    checkpoint.conditional_dict = to_device(checkpoint.conditional_dict, device)
    for name in ("latent_history", "previous_chunk_latent", "next_noise"):
        value = getattr(checkpoint, name)
        if isinstance(value, torch.Tensor):
            setattr(checkpoint, name, value.to(device=device))
    if checkpoint.denoise_noise is not None:
        checkpoint.denoise_noise = [value.to(device=device) for value in checkpoint.denoise_noise]
    return checkpoint


def checkpoint_manifest_entry(path: os.PathLike | str, checkpoint: EditCheckpoint,
                              **extra: Any) -> Dict[str, Any]:
    path = Path(path)
    entry = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size,
             "sample_id": checkpoint.sample_id, "chunk_index": checkpoint.chunk_index,
             "cache_bytes": checkpoint.cache_bytes(), "identity": checkpoint.identity()}
    entry.update(extra)
    return entry


# --------------------------------------------------------------------------- #
# RNG state
# --------------------------------------------------------------------------- #
def capture_rng_state(device: torch.device | str | None = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {"torch_rng_state": torch.get_rng_state().clone(),
                             "cuda_rng_state": []}
    if torch.cuda.is_available():
        devices: Sequence[int]
        if device is None:
            devices = range(torch.cuda.device_count())
        else:
            devices = [torch.device(device).index or 0]
        state["cuda_rng_state"] = [torch.cuda.get_rng_state(index).clone() for index in devices]
    return state


def restore_rng_state(torch_rng_state: torch.Tensor, cuda_rng_state: Sequence[torch.Tensor],
                      device: torch.device | str | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG state cannot be restored on a CPU-only host")
    torch.set_rng_state(torch_rng_state.to("cpu"))
    index = 0 if device is None else (torch.device(device).index or 0)
    if index >= len(cuda_rng_state):
        raise ValueError(f"No CUDA RNG state recorded for device {index}")
    torch.cuda.set_rng_state(cuda_rng_state[index].to("cpu"), index)


# --------------------------------------------------------------------------- #
# capture / restore against a live pipeline
# --------------------------------------------------------------------------- #
def capture_kv_cache(pipeline) -> List[Dict[str, Any]]:
    if pipeline.kv_cache1 is None:
        return []
    return [{"k": block["k"].detach().clone(), "v": block["v"].detach().clone(),
             "global_end_index": block["global_end_index"].detach().clone(),
             "local_end_index": block["local_end_index"].detach().clone()}
            for block in pipeline.kv_cache1]


def capture_crossattn_cache(pipeline) -> List[Dict[str, Any]]:
    if getattr(pipeline, "crossattn_cache", None) is None:
        return []
    blocks = []
    for block in pipeline.crossattn_cache:
        entry: Dict[str, Any] = {"is_init": bool(block["is_init"])}
        for name in ("k", "v"):
            value = block.get(name)
            entry[name] = value.detach().clone() if isinstance(value, torch.Tensor) else value
        blocks.append(entry)
    return blocks


def scheduler_provenance(pipeline) -> Dict[str, Any]:
    scheduler = pipeline.scheduler
    record: Dict[str, Any] = {"class": type(scheduler).__name__}
    for name in ("timesteps", "sigmas"):
        value = getattr(scheduler, name, None)
        if isinstance(value, torch.Tensor):
            record[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": hashlib.sha256(value.float().cpu().numpy().tobytes()).hexdigest()}
    record["denoising_step_list"] = [int(v) for v in pipeline.denoising_step_list.tolist()]
    record["context_noise"] = int(getattr(pipeline.args, "context_noise", 0))
    return record


def position_provenance(pipeline, current_start_frame: int) -> Dict[str, Any]:
    model = pipeline.generator.model
    freqs = getattr(model, "freqs", None)
    return {"current_start_frame": int(current_start_frame),
            "current_start_token": int(current_start_frame) * int(pipeline.frame_seq_length),
            "frame_seq_length": int(pipeline.frame_seq_length),
            "local_attn_size": int(pipeline.local_attn_size),
            "max_attention_size": int(getattr(model, "max_attention_size", -1)),
            "rope_freqs": None if freqs is None else {"shape": list(freqs.shape),
                                                      "dtype": str(freqs.dtype)}}


def capture_checkpoint(
    pipeline,
    *,
    sample_id: str,
    chunk_index: int,
    num_chunks: int,
    prompt: str,
    seed: int,
    model_hash: str,
    model_record: Dict[str, Any],
    config_digest: str,
    conditional_dict: Optional[Dict[str, Any]],
    latent_history: Optional[torch.Tensor],
    current_start_frame: int,
    full_latent_shape: Optional[Sequence[int]] = None,
    previous_chunk_latent: Optional[torch.Tensor] = None,
    next_noise: Optional[torch.Tensor] = None,
    denoise_noise: Optional[List[torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str | None = None,
) -> EditCheckpoint:
    """Snapshot the state that sits *before* chunk ``chunk_index`` is denoised."""
    rng = capture_rng_state(device)
    generator_state = generator.get_state().clone() if generator is not None else None
    if full_latent_shape is not None:
        latent_shape = [int(v) for v in full_latent_shape]
    elif isinstance(latent_history, torch.Tensor):
        latent_shape = list(latent_history.shape)
    else:
        latent_shape = []
    checkpoint = EditCheckpoint(
        schema_version=SCHEMA_VERSION,
        sample_id=sample_id,
        chunk_index=int(chunk_index),
        num_chunks=int(num_chunks),
        model_hash=model_hash,
        model_identity=model_record,
        config_hash=config_digest,
        prompt=prompt,
        prompt_hash=sha256_text(prompt),
        seed=int(seed),
        code_commit=code_commit(),
        created_unix=time.time(),
        num_frame_per_block=int(pipeline.num_frame_per_block),
        frame_seq_length=int(pipeline.frame_seq_length),
        local_attn_size=int(pipeline.local_attn_size),
        sink_size=int(getattr(pipeline.args.model_kwargs, "sink_size", 0)),
        latent_shape=latent_shape,
        latent_dtype=str(pipeline.generator.model.patch_embedding.weight.dtype),
        source_device=str(device if device is not None else "unknown"),
        current_start_frame=int(current_start_frame),
        torch_rng_state=rng["torch_rng_state"],
        cuda_rng_state=rng["cuda_rng_state"],
        generator_state=generator_state,
        latent_history=latent_history,
        kv_cache=capture_kv_cache(pipeline),
        crossattn_cache=capture_crossattn_cache(pipeline),
        sink_cache=None,  # sinks live inside kv_cache1 for this backbone
        position_state=position_provenance(pipeline, current_start_frame),
        scheduler_state=scheduler_provenance(pipeline),
        conditional_dict=conditional_dict,
        previous_chunk_latent=previous_chunk_latent,
        next_noise=next_noise,
        denoise_noise=denoise_noise,
    )
    checkpoint.provenance = {
        "tensors": tensor_provenance(checkpoint.to_payload()),
        "cache_bytes": checkpoint.cache_bytes(),
        "crossattn_initialised_blocks": sum(bool(b["is_init"]) for b in checkpoint.crossattn_cache)
        if checkpoint.crossattn_cache else 0,
        "kv_blocks": len(checkpoint.kv_cache),
    }
    return checkpoint


def restore_checkpoint(pipeline, checkpoint: EditCheckpoint,
                       device: torch.device | str | None = None,
                       reset_crossattn: bool = False) -> Dict[str, Any]:
    """Install ``checkpoint`` into a live pipeline.

    ``reset_crossattn=True`` clears the text cross-attention cache so the next
    forward recomputes K/V from the prompt embeds it is given, which is exactly
    the mechanism a new prompt needs.  ``reset_crossattn=False`` restores the
    original prompt binding bit-for-bit.
    """
    if not checkpoint.kv_cache:
        raise ValueError("Checkpoint has no KV cache; cannot restore generation state")
    target = device if device is not None else checkpoint.source_device
    # Always build fresh device tensors: replay must never alias (and therefore
    # mutate) the checkpoint it was restored from, so the same checkpoint can be
    # replayed any number of times.
    kv_cache = [_copy_block(block, target) for block in checkpoint.kv_cache]
    pipeline.kv_cache1 = list(kv_cache)
    if checkpoint.crossattn_cache:
        pipeline.crossattn_cache = [_copy_block(block, target) for block in checkpoint.crossattn_cache]
    elif getattr(pipeline, "crossattn_cache", None) is None:
        raise ValueError("Checkpoint has no cross-attention cache and pipeline has none allocated")
    if reset_crossattn:
        reset_crossattn_cache(pipeline)
    if checkpoint.latent_shape:
        pipeline.generator.model.local_attn_size = int(pipeline.local_attn_size)
        pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)
    return {"current_start_frame": int(checkpoint.current_start_frame),
            "kv_blocks": len(pipeline.kv_cache1),
            "crossattn_reset": bool(reset_crossattn)}


def _copy_block(block: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    """Fresh device copy of one KV / cross-attention cache block."""
    return {key: (value.to(device=device).clone() if isinstance(value, torch.Tensor) else value)
            for key, value in block.items()}


def reset_crossattn_cache(pipeline) -> None:
    """Clear text K/V so the next forward rebinds to a new prompt."""
    if getattr(pipeline, "crossattn_cache", None) is None:
        raise ValueError("Pipeline has no cross-attention cache allocated")
    for block in pipeline.crossattn_cache:
        block["is_init"] = False
        for name in ("k", "v"):
            value = block.get(name)
            if isinstance(value, torch.Tensor):
                value.zero_()


def verify_checkpoint_identity(checkpoint: EditCheckpoint, *, model_hash: Optional[str] = None,
                               config_digest: Optional[str] = None,
                               num_frame_per_block: Optional[int] = None,
                               frame_seq_length: Optional[int] = None,
                               local_attn_size: Optional[int] = None,
                               latent_shape: Optional[Sequence[int]] = None) -> None:
    """Refuse to silently reuse a cache produced by different settings."""
    if model_hash is not None and checkpoint.model_hash != model_hash:
        raise ValueError("Checkpoint was produced by a different model")
    if config_digest is not None and checkpoint.config_hash != config_digest:
        raise ValueError("Checkpoint was produced by a different generation config")
    if num_frame_per_block is not None and checkpoint.num_frame_per_block != num_frame_per_block:
        raise ValueError("Checkpoint AR block size differs from the live pipeline")
    if frame_seq_length is not None and checkpoint.frame_seq_length != frame_seq_length:
        raise ValueError("Checkpoint token geometry differs from the live pipeline")
    if local_attn_size is not None and checkpoint.local_attn_size != local_attn_size:
        raise ValueError("Checkpoint attention window differs from the live pipeline")
    if latent_shape is not None and list(latent_shape) != list(checkpoint.latent_shape):
        raise ValueError("Checkpoint latent shape differs from the requested generation")
