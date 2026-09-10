"""Streaming generation with chunk-boundary edit checkpoints.

This mirrors ``CausalInferencePipeline.inference`` exactly (same cache sizing, same
denoising schedule, same cache re-entry with ``context_noise``) but exposes the AR
chunk loop so that:

* a checkpoint can be captured *before* every chunk (``stream_generate``),
* one chunk can be reopened from a checkpoint afterwards (``replay_chunk``),
* the reopened chunk can be spliced back with bit-exact outside preservation
  (``assemble_latents``).

Use ``verify_upstream_equivalence`` to prove the mirrored loop still matches the
upstream pipeline bit-for-bit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

from .edit_cache import (
    EditCheckpoint,
    capture_checkpoint,
    checkpoint_to_device,
    restore_checkpoint,
    restore_rng_state,
    save_edit_checkpoint,
)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def pixel_frames_to_latents(pixel_frames: int) -> int:
    if (pixel_frames - 1) % 4:
        raise ValueError("Pixel frame count must be 4k+1")
    return (pixel_frames - 1) // 4 + 1


def latent_shape(config: Dict[str, Any]) -> Sequence[int]:
    frames = config["generation"].get("num_latent_frames")
    if frames is None:
        frames = pixel_frames_to_latents(int(config["generation"]["pixel_frames"]))
    height = int(config["generation"]["height"]) // 8
    width = int(config["generation"]["width"]) // 8
    return (1, int(frames), 16, height, width)


def num_chunks_for(config: Dict[str, Any], pipeline) -> int:
    shape = latent_shape(config)
    block = int(pipeline.num_frame_per_block)
    if shape[1] % block:
        raise ValueError(f"num_latent_frames={shape[1]} must be a multiple of num_frame_per_block={block}")
    return shape[1] // block


# --------------------------------------------------------------------------- #
# conditioning
# --------------------------------------------------------------------------- #
@torch.no_grad()
def make_conditioning(pipeline, prompt: str) -> Dict[str, Any]:
    return pipeline.text_encoder(text_prompts=[prompt])


# --------------------------------------------------------------------------- #
# cache allocation (mirrors CausalInferencePipeline.inference step 1)
# --------------------------------------------------------------------------- #
def initialize_streaming_state(pipeline, noise: torch.Tensor) -> None:
    batch_size, num_output_frames = noise.shape[:2]
    local_attn_cfg = getattr(pipeline.args.model_kwargs, "local_attn_size", -1)
    if local_attn_cfg != -1:
        kv_cache_size = int(local_attn_cfg) * int(pipeline.frame_seq_length)
    else:
        kv_cache_size = int(num_output_frames) * int(pipeline.frame_seq_length)
    pipeline._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device,
                                  kv_cache_size_override=kv_cache_size)
    pipeline._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
    pipeline.generator.model.local_attn_size = pipeline.local_attn_size
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)


def allocate_like(pipeline, kv_cache: Sequence[Dict[str, Any]], crossattn_cache: Sequence[Dict[str, Any]]) -> None:
    """Install zeroed caches with the geometry of a checkpoint.

    Only needed when a checkpoint is inspected without being restored; the normal
    replay path goes through :func:`restream.edit_cache.restore_checkpoint`, which
    builds the caches directly from the checkpoint.
    """
    pipeline.kv_cache1 = [{"k": torch.zeros_like(block["k"]), "v": torch.zeros_like(block["v"]),
                           "global_end_index": torch.zeros_like(block["global_end_index"]),
                           "local_end_index": torch.zeros_like(block["local_end_index"])}
                          for block in kv_cache]
    pipeline.crossattn_cache = [{"k": torch.zeros_like(block["k"]), "v": torch.zeros_like(block["v"]),
                                 "is_init": False}
                                for block in crossattn_cache]


# --------------------------------------------------------------------------- #
# chunk denoising (mirrors CausalInferencePipeline.inference step 2)
# --------------------------------------------------------------------------- #
def _draw_noise(reference: torch.Tensor, mode: str, generator: Optional[torch.Generator],
                stored: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mode == "global":
        return torch.randn_like(reference)
    if mode == "generator":
        if generator is None:
            raise ValueError("noise_source='generator' requires a torch.Generator")
        return torch.randn(reference.shape, generator=generator, device=reference.device,
                           dtype=reference.dtype)
    if mode == "stored":
        if stored is None:
            raise ValueError("noise_source='stored' requires stored noise")
        return stored.to(device=reference.device, dtype=reference.dtype)
    raise ValueError(f"Unknown noise source {mode!r}")


def denoise_chunk(pipeline, noisy_input: torch.Tensor, conditional_dict: Dict[str, Any],
                  current_start_frame: int, *, noise_source: str = "global",
                  generator: Optional[torch.Generator] = None,
                  stored_noise: Optional[Sequence[torch.Tensor]] = None,
                  record_noise: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
    """Run the spatial denoising loop for one AR chunk.

    ``noise_source``:
      * ``global``    - ``torch.randn_like``, byte-identical to upstream inference;
      * ``generator`` - explicit ``torch.Generator`` (deterministic replay without
                        touching global RNG state);
      * ``stored``    - reuse the in-chunk noise draws recorded in a checkpoint.
    """
    batch_size, current_num_frames = noisy_input.shape[:2]
    steps = pipeline.denoising_step_list
    denoised_pred = noisy_input
    for index, current_timestep in enumerate(steps):
        timestep = torch.ones([batch_size, current_num_frames], device=noisy_input.device,
                              dtype=torch.int64) * current_timestep
        # Feed the *current* (re-noised) latent, exactly like upstream inference.
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )
        if index < len(steps) - 1:
            flat = denoised_pred.flatten(0, 1)
            stored = None if stored_noise is None else stored_noise[index]
            extra = _draw_noise(flat, noise_source, generator, stored)
            if record_noise is not None:
                record_noise.append(extra.detach().to("cpu", copy=True))
            next_timestep = steps[index + 1]
            noisy_input = pipeline.scheduler.add_noise(
                flat,
                extra,
                next_timestep * torch.ones([batch_size * current_num_frames], device=noisy_input.device,
                                           dtype=torch.long),
            ).unflatten(0, denoised_pred.shape[:2])
    return _context_update(pipeline, denoised_pred, conditional_dict, current_start_frame)


def _context_update(pipeline, denoised_pred: torch.Tensor, conditional_dict: Dict[str, Any],
                    current_start_frame: int) -> torch.Tensor:
    """Re-run the chunk with ``context_noise`` so clean context enters the KV cache."""
    batch_size, current_num_frames = denoised_pred.shape[:2]
    context_timestep = torch.ones([batch_size, current_num_frames], device=denoised_pred.device,
                                  dtype=torch.int64) * int(pipeline.args.context_noise)
    pipeline.generator(
        noisy_image_or_video=denoised_pred,
        conditional_dict=conditional_dict,
        timestep=context_timestep,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start_frame * pipeline.frame_seq_length,
    )
    return denoised_pred


# --------------------------------------------------------------------------- #
# full streaming generation with checkpoints
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    latents: torch.Tensor
    sample_id: str
    prompt: str
    noise: torch.Tensor
    checkpoint_entries: Dict[int, Dict[str, Any]]
    timings: Dict[str, float]
    peak_vram_bytes: int


@torch.no_grad()
def stream_generate(pipeline, noise: torch.Tensor, prompt: str, *, sample_id: str = "sample",
                    seed: int = 0, model_hash: str = "", model_record: Optional[Dict[str, Any]] = None,
                    config_digest: str = "", cache_dir: Optional[Path] = None,
                    save_chunks: Optional[Iterable[int]] = None, noise_source: str = "global",
                    generator: Optional[torch.Generator] = None,
                    on_chunk: Optional[Callable[[int, torch.Tensor], None]] = None,
                    time_it: bool = True) -> GenerationResult:
    """Generate the whole video, capturing ``EditCheckpoint`` before each saved chunk."""
    batch_size, num_frames = noise.shape[:2]
    block = int(pipeline.num_frame_per_block)
    if num_frames % block:
        raise ValueError("Latent frame count must be a multiple of the AR block size")
    num_chunks = num_frames // block
    save_set = set(range(num_chunks)) if save_chunks is None else {int(k) for k in save_chunks}

    conditioning = make_conditioning(pipeline, prompt)
    initialize_streaming_state(pipeline, noise)
    latents = torch.zeros_like(noise)
    entries: Dict[int, Dict[str, Any]] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(noise.device)
    started = time.perf_counter()
    for index in range(num_chunks):
        start = index * block
        noisy_input = noise[:, start:start + block]
        checkpoint: Optional[EditCheckpoint] = None
        if index in save_set:
            checkpoint = capture_checkpoint(
                pipeline,
                sample_id=sample_id,
                chunk_index=index,
                num_chunks=num_chunks,
                prompt=prompt,
                seed=seed,
                model_hash=model_hash,
                model_record=model_record or {},
                config_digest=config_digest,
                conditional_dict=conditioning,
                latent_history=latents[:, :start].clone(),
                current_start_frame=start,
                full_latent_shape=noise.shape,
                previous_chunk_latent=None if start == 0 else latents[:, start - block:start].clone(),
                next_noise=noise[:, start:start + block].clone(),
            )
        recorded: List[torch.Tensor] = []
        denoised = denoise_chunk(pipeline, noisy_input, conditioning, start,
                                 noise_source=noise_source, generator=generator,
                                 record_noise=recorded)
        latents[:, start:start + block] = denoised
        if checkpoint is not None:
            checkpoint.denoise_noise = recorded or None
            if cache_dir is not None:
                path = Path(cache_dir) / f"{sample_id}__chunk{index:03d}.pt"
                entries[index] = save_edit_checkpoint(path, checkpoint)
            else:
                entries[index] = {"chunk_index": index, "sample_id": sample_id,
                                  "checkpoint": checkpoint,
                                  "cache_bytes": checkpoint.cache_bytes()}
        if on_chunk is not None:
            on_chunk(index, denoised)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(noise.device) if torch.cuda.is_available() else 0
    return GenerationResult(latents=latents, sample_id=sample_id, prompt=prompt, noise=noise,
                            checkpoint_entries=entries,
                            timings={"full_generation_seconds": elapsed} if time_it else {},
                            peak_vram_bytes=int(peak))


# --------------------------------------------------------------------------- #
# replay / edit of a single chunk
# --------------------------------------------------------------------------- #
@dataclass
class ReplayResult:
    latents: torch.Tensor            # [1, block, C, H, W] reopened chunk
    prompt: str
    chunk_index: int
    current_start_frame: int
    recorded_noise: Optional[List[torch.Tensor]]
    timings: Dict[str, float]
    peak_vram_bytes: int


@torch.no_grad()
def replay_chunk(pipeline, checkpoint: EditCheckpoint, prompt: str, *,
                 device: torch.device | str | None = None,
                 noise: str = "from-cache", noise_source: str = "global",
                 crossattn: str = "restore", restore_rng: bool = True,
                 generator: Optional[torch.Generator] = None,
                 allocate: bool = False, time_it: bool = True) -> ReplayResult:
    """Reopen exactly one chunk from ``checkpoint``.

    ``noise``:
      * ``from-cache`` - start from the exact original chunk noise slice saved in
        the checkpoint (the only way to reproduce the initial noise, which was
        drawn before the chunk boundary);
      * ``draw``       - draw the initial noise from the restored RNG state
        (diagnostic: shows how much of the output is noise-driven).
    ``noise_source`` selects the engine for the *in-chunk* re-noising draws
    (``global`` reproduces upstream; ``stored`` reuses the recorded draws).
    ``crossattn``:
      * ``reset``   - clear text K/V so a new prompt rebinds (this is the edit);
      * ``restore`` - keep the original text K/V.  Combined with a *new* prompt
        this is the no-op control that isolates "prompt changed" from "text
        binding changed": the new prompt never enters the computation.
    """
    if not checkpoint.kv_cache:
        raise ValueError("Checkpoint has no KV cache to replay from")
    if device is None:
        device = pipeline.generator.model.patch_embedding.weight.device
    if allocate:
        allocate_like(pipeline, checkpoint.kv_cache, checkpoint.crossattn_cache or [])
    started = time.perf_counter()
    recorded: List[torch.Tensor] = []
    restore_checkpoint(pipeline, checkpoint, device=device,
                       reset_crossattn=(crossattn == "reset"))
    if restore_rng:
        restore_rng_state(checkpoint.torch_rng_state, checkpoint.cuda_rng_state, device)
    if generator is not None and checkpoint.generator_state is not None:
        generator.set_state(checkpoint.generator_state.to("cpu"))
    if prompt == checkpoint.prompt and checkpoint.conditional_dict is not None:
        conditioning = checkpoint.conditional_dict
    else:
        conditioning = make_conditioning(pipeline, prompt)
    conditioning = _conditioning_to_device(conditioning, device)
    restore_seconds = time.perf_counter() - started

    if noise == "from-cache":
        if checkpoint.next_noise is None:
            raise ValueError("Checkpoint has no saved chunk noise; use noise='draw'")
        initial = checkpoint.next_noise.to(device=device)
    elif noise == "draw":
        if checkpoint.next_noise is None:
            raise ValueError("Checkpoint has no noise shape reference")
        initial = torch.randn(checkpoint.next_noise.shape, device=device,
                              dtype=checkpoint.next_noise.dtype)
    else:
        raise ValueError(f"Unknown replay noise mode {noise!r}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    denoise_started = time.perf_counter()
    latents = denoise_chunk(pipeline, initial, conditioning, checkpoint.current_start_frame,
                            noise_source=noise_source, generator=generator,
                            stored_noise=checkpoint.denoise_noise,
                            record_noise=recorded)
    denoise_seconds = time.perf_counter() - denoise_started
    peak = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    return ReplayResult(latents=latents, prompt=prompt, chunk_index=checkpoint.chunk_index,
                        current_start_frame=checkpoint.current_start_frame,
                        recorded_noise=recorded or None,
                        timings=({"restore_seconds": restore_seconds,
                                  "denoise_seconds": denoise_seconds} if time_it else {}),
                        peak_vram_bytes=int(peak))


def _conditioning_to_device(conditioning: Dict[str, Any], device) -> Dict[str, Any]:
    return {key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in conditioning.items()}


def verify_upstream_equivalence(pipeline, noise: torch.Tensor, prompt: str) -> Dict[str, Any]:
    """Compare the mirrored loop with ``CausalInferencePipeline.inference``.

    Both runs are seeded identically; the upstream path returns latents when
    ``return_latents=True``.  Any mismatch means the mirrored loop drifted.
    """
    seed = torch.initial_seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _, upstream_latents = pipeline.inference(noise=noise.clone(), text_prompts=[prompt],
                                             return_latents=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    result = stream_generate(pipeline, noise.clone(), prompt, sample_id="equivalence",
                             config_digest="", model_hash="", save_chunks=[])
    a = upstream_latents.float()
    b = result.latents.float()
    return {"exact": bool(torch.equal(upstream_latents, result.latents)),
            "max_abs_diff": float((a - b).abs().max().item()),
            "mse": float((a - b).square().mean().item()),
            "cosine": float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())}


# --------------------------------------------------------------------------- #
# assembly, decoding, metrics plumbing
# --------------------------------------------------------------------------- #
def assemble_latents(base_latents: torch.Tensor, edited_chunk: torch.Tensor, chunk_index: int,
                     num_frame_per_block: int) -> torch.Tensor:
    """Splice ``edited_chunk`` into ``base_latents`` and assert exact outside preservation."""
    block = int(num_frame_per_block)
    start = int(chunk_index) * block
    if edited_chunk.shape[1] != block:
        raise ValueError(f"Edited chunk has {edited_chunk.shape[1]} frames, expected {block}")
    if base_latents.shape[1] % block:
        raise ValueError("Base latents do not align to AR blocks")
    assembled = base_latents.clone()
    assembled[:, start:start + block] = edited_chunk.to(assembled.dtype)
    outside_exact(assembled, base_latents, chunk_index, block)
    return assembled


def outside_exact(candidate: torch.Tensor, reference: torch.Tensor, chunk_index: int,
                  num_frame_per_block: int) -> int:
    """Hard ``torch.equal`` check on every chunk outside ``chunk_index``."""
    block = int(num_frame_per_block)
    checked = 0
    for index in range(reference.shape[1] // block):
        if index == chunk_index:
            continue
        start = index * block
        if not torch.equal(candidate[:, start:start + block], reference[:, start:start + block]):
            raise AssertionError(
                f"Outside preservation violated at chunk {index} (edited chunk {chunk_index})")
        checked += 1
    return checked


@torch.no_grad()
def decode_latents(pipeline, latents: torch.Tensor) -> torch.Tensor:
    """Latent -> pixel in [0, 1]; identical to upstream inference step 3."""
    if getattr(pipeline.args.model_kwargs, "use_infinite_attention", False):
        video = pipeline.vae.decode_to_pixel_chunk(latents.to(latents.device), use_cache=False)
    else:
        video = pipeline.vae.decode_to_pixel(latents.to(latents.device), use_cache=False)
    return (video * 0.5 + 0.5).clamp(0, 1)


def chunk_frame_slice(chunk_index: int, num_frame_per_block: int, pixel_frames: Optional[int] = None):
    """Latent/pixel frame range covered by an AR chunk.

    Wan maps ``L`` latent frames to ``4(L-1)+1`` pixel frames with a causal VAE:
    the first latent emits one frame and every later latent emits four.  Chunk
    ``k`` covers latents ``[3k, 3k+3)``, i.e. pixels ``[4(3k-1)+1, 4(3k+2)+1)``
    for ``k >= 1`` and ``[0, 9)`` for ``k == 0``.  The ranges tile the video
    exactly with no gaps.
    """
    block = int(num_frame_per_block)
    latent_start = chunk_index * block
    latent_end = latent_start + block
    pixel_start = 0 if latent_start == 0 else 4 * (latent_start - 1) + 1
    pixel_end = 4 * (latent_end - 1) + 1
    if pixel_frames is not None:
        pixel_end = min(pixel_end, int(pixel_frames))
    return {"latent_start": latent_start, "latent_end": latent_end,
            "pixel_start": pixel_start, "pixel_end": pixel_end}
