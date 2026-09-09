"""Precompute frozen reference features. Never runs the video generator or an optimizer."""
import argparse
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import read_manifest
from restream.reality_data import read_reference, write_json
from restream.reality_encoder import RealityEncoder
from restream.reality_runtime import read_reality_config, make_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/reality_memory_r0.yaml")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = read_reality_config(args.config)
    memory = config["reality_memory"]
    cache = make_cache(config)
    refs = {}
    for split in ("train", "val"):
        for row in read_manifest(ROOT / config["data"][f"{split}_manifest"]):
            selected = row["references"] if split == "train" else [ref for pool in row["reference_sets"].values() for ref in pool]
            for ref in selected:
                refs[cache.key(ref)] = ref
    encoder = RealityEncoder(ROOT / memory["encoder"]["path"], memory["projector"]["memory_dim"],
                             memory["projector"]["num_memory_tokens"], memory["encoder"]["image_size"]).to(args.device).eval()
    written, reused = 0, 0
    for i, ref in enumerate(refs.values()):
        if cache.path(ref).exists():
            cache.read(ref)  # Shape, finite values and identity check on reuse.
            reused += 1
        else:
            pixels = torch.from_numpy(read_reference(ref)).permute(2, 0, 1).float() / 255
            features = encoder.encode_visual(pixels[None, None])[0, 0]
            cache.write(ref, features)
            written += 1
        if (i + 1) % 100 == 0:
            print(f"Features {i + 1}/{len(refs)}", flush=True)
    report = {"unique_references": len(refs), "written": written, "reused": reused, "encoder": cache.identity,
              "shape_per_reference": [cache.tokens, cache.channels], "optimizer_steps": 0}
    write_json(ROOT / "data/reality_feature_stats.json", report)
    print(report)


if __name__ == "__main__":
    main()
