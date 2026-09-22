"""LongLive v1.0 integration. Heavy dependencies are imported only on demand."""
import os
import sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def load_pipeline(config, device):
    upstream = ROOT / "code/LongLive"
    sys.path.insert(0, str(upstream))
    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from utils.lora_utils import configure_lora_for_model
    import peft
    base = OmegaConf.load(upstream / "configs/default_config.yaml")
    args = OmegaConf.merge(base, OmegaConf.load(ROOT / config["model"]["upstream_config"]))
    directory = ROOT / config["model"]["longlive"]

    def locate(name):
        matches = list(directory.rglob(name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one official {name} in {directory}; got {matches}")
        return matches[0]

    checkpoint = locate(Path(args.generator_ckpt).name)
    lora_checkpoint = locate(Path(args.lora_ckpt).name)
    link = upstream / "wan_models/Wan2.1-T2V-1.3B"
    link.parent.mkdir(exist_ok=True)
    target = (ROOT / config["model"]["wan"]).resolve()
    if not link.exists():
        try:
            link.symlink_to(target, target_is_directory=True)
        except FileExistsError:
            pass  # another DDP rank created it
    if link.resolve() != target:
        raise RuntimeError("Existing Wan model link points to another installation")
    old = Path.cwd()
    try:
        os.chdir(upstream)  # upstream wrappers use hardcoded relative model paths
        pipeline = CausalInferencePipeline(args, device)
    finally:
        os.chdir(old)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    raw = state.get("generator", state.get("model"))
    if raw is None:
        raise ValueError("Official checkpoint has no generator/model key")
    pipeline.generator.load_state_dict(raw, strict=True)
    pipeline.generator.model = configure_lora_for_model(pipeline.generator.model, "generator", args.adapter)
    lora = torch.load(lora_checkpoint, map_location="cpu", weights_only=True)
    weights = lora.get("generator_lora", lora)
    # Validate that every expected adapter tensor is provided; PEFT's report also
    # lists missing frozen backbone tensors, so compare adapter states directly.
    expected = peft.get_peft_model_state_dict(pipeline.generator.model)
    if set(expected) != set(weights) or any(expected[k].shape != weights[k].shape for k in expected):
        raise ValueError("Official LoRA keys or tensor shapes do not match the configured baseline")
    peft.set_peft_model_state_dict(pipeline.generator.model, weights)
    pipeline.eval().requires_grad_(False)
    placements = config["model"].get("component_devices")
    if placements:
        pipeline.to(dtype=torch.bfloat16)
        pipeline.text_encoder.to(placements["text_encoder"])
        pipeline.vae.to(placements["vae"])
        pipeline.generator.to(device)
    else:
        pipeline.to(device=device, dtype=torch.bfloat16)
    # Cache token offsets must reflect actual spatial resolution (upstream: 1560).
    h, w = config["data"]["height"] // 8, config["data"]["width"] // 8
    pipeline.frame_seq_length = (h // 2) * (w // 2)
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)
    return pipeline


def new_cache(pipeline, history, total_frames):
    size = total_frames if pipeline.local_attn_size == -1 else pipeline.local_attn_size
    pipeline._initialize_kv_cache(history.shape[0], history.dtype, history.device,
                                  kv_cache_size_override=size * pipeline.frame_seq_length)
    pipeline._initialize_crossattn_cache(history.shape[0], history.dtype, history.device)


def cache_block(pipeline, block, conditioning, start):
    return pipeline.generator(block, conditioning,
                              torch.zeros(block.shape[:2], device=block.device, dtype=torch.long),
                              kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                              current_start=start * pipeline.frame_seq_length)


@torch.no_grad()
def rollout(pipeline, prefix, conditioning, noise, generator):
    """Fallback B: fresh cache, replay all recent corrected history, continue AR."""
    block = pipeline.num_frame_per_block
    if prefix.shape[1] % block or noise.shape[1] % block:
        raise ValueError("Prefix and future must end at AR block boundaries")
    new_cache(pipeline, prefix, prefix.shape[1] + noise.shape[1])
    for start in range(0, prefix.shape[1], block):
        cache_block(pipeline, prefix[:, start:start + block], conditioning, start)
    outputs = [prefix]
    for offset in range(0, noise.shape[1], block):
        noisy = noise[:, offset:offset + block]
        start = prefix.shape[1] + offset
        for i, time in enumerate(pipeline.denoising_step_list):
            timestep = torch.full(noisy.shape[:2], float(time), device=noisy.device)
            _, clean = pipeline.generator(noisy, conditioning, timestep,
                                          kv_cache=pipeline.kv_cache1, crossattn_cache=pipeline.crossattn_cache,
                                          current_start=start * pipeline.frame_seq_length)
            if i + 1 < len(pipeline.denoising_step_list):
                next_time = pipeline.denoising_step_list[i + 1].to(noisy.device)
                extra = torch.randn(clean.shape, generator=generator, device=clean.device, dtype=clean.dtype)
                noisy = pipeline.scheduler.add_noise(clean.flatten(0, 1), extra.flatten(0, 1),
                                                      next_time.expand(clean.shape[0] * block)).unflatten(0, clean.shape[:2])
        outputs.append(clean)
        cache_block(pipeline, clean, conditioning, start)
    return torch.cat(outputs, dim=1)
