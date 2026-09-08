"""Preparation check: load frozen assets and verify VAE layout; no generation/SFT."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from restream.runtime import ROOT, read_config, load_pipeline
from restream.anchor_injector import HardAnchorInjector

torch.cuda.set_device(0)
config = read_config(ROOT / "configs/restream_mvp.yaml")
pipeline = load_pipeline(config, torch.device("cuda", 0))
with torch.no_grad():
    pixels = torch.zeros(1, 3, 9, 64, 64, device="cuda", dtype=torch.bfloat16)
    video_latent = pipeline.vae.encode_to_latent(pixels)
    image_latent = HardAnchorInjector(pipeline.vae).encode_anchor(pixels[:, :, 0])
assert video_latent.shape == (1, 3, 16, 8, 8), video_latent.shape
assert image_latent.shape == (1, 1, 16, 8, 8), image_latent.shape
assert torch.isfinite(video_latent).all() and torch.isfinite(image_latent).all()
assert not any(p.requires_grad for p in pipeline.parameters())
report = {"checkpoint_load": "strict keys and LoRA tensor shapes passed",
          "video_latent_shape": list(video_latent.shape), "image_latent_shape": list(image_latent.shape),
          "all_backbone_parameters_frozen": True, "training_steps": 0,
          "gpu": torch.cuda.get_device_name(0), "peak_vram_bytes": torch.cuda.max_memory_allocated()}
(ROOT / "logs/model_load_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
