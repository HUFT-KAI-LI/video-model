"""Real frozen DINO feature extraction/cache/projector check; zero optimizer updates."""
import argparse
from pathlib import Path
import sys
import tempfile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import read_manifest
from restream.reality_cache import FeatureCache
from restream.reality_data import read_reference, write_json
from restream.reality_encoder import RealityEncoder
from restream.reality_runtime import read_reality_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/reality_memory/encoder_check.json")
    args = parser.parse_args()
    config = read_reality_config(args.config)
    cfg = config["reality_memory"]
    torch.manual_seed(config["seed"])
    encoder = RealityEncoder(ROOT / cfg["encoder"]["path"], cfg["projector"]["memory_dim"],
                             cfg["projector"]["num_memory_tokens"], cfg["encoder"]["image_size"]).to(args.device)
    row = next(row for row in read_manifest(ROOT / config["data"]["train_manifest"]) if row["references"])
    ref = row["references"][0]
    pixels = torch.from_numpy(read_reference(ref)).permute(2, 0, 1).float()[None, None] / 255
    raw = encoder.encode_visual(pixels)
    assert raw.shape == (1, 1, cfg["projector"]["num_memory_tokens"], cfg["encoder"]["feature_dim"])
    assert not raw.requires_grad and torch.isfinite(raw).all()
    output = encoder.project(raw)
    output.square().mean().backward()
    assert all(p.grad is None for p in encoder.backbone.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.projector.parameters())
    with tempfile.TemporaryDirectory() as folder:
        cache = FeatureCache(folder, encoder.identity, raw.shape[2], raw.shape[3])
        cache.write(ref, raw[0, 0])
        assert torch.equal(cache.read(ref), raw[0, 0].cpu())
    report = {"status": "passed", "source_id": ref["source_id"], "raw_shape": list(raw.shape),
              "projected_shape": list(output.shape), "frozen_backbone": True, "cache_exact_reload": True,
              "encoder": encoder.identity, "optimizer_steps": 0,
              "projector_grad_norm": sum(p.grad.float().square().sum() for p in encoder.projector.parameters()).sqrt().item()}
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
