"""Download official weights; never starts inference or training."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--model", choices=["wan", "longlive", "longlive-modelscope"], required=True)
    args = p.parse_args()
    if args.model == "wan":
        from modelscope.hub.snapshot_download import snapshot_download
        dest = args.root / "models/Wan2.1-T2V-1.3B"
        snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=str(dest), max_workers=4)
        source = "modelscope:Wan-AI/Wan2.1-T2V-1.3B"
    elif args.model == "longlive-modelscope":
        from modelscope.hub.snapshot_download import snapshot_download
        dest = args.root / "models/LongLive-1.3B"
        snapshot_download("Efficient-Large-Model/LongLive-1.3B", local_dir=str(dest), max_workers=2,
                          allow_patterns=["models/*.pt", "README.md", "configuration.json"])
        expected = {
            "models/longlive_base.pt": "10a2aa8fcf89c77d9033f4c117405412a690e289625766619d293f0c5a208ee7",
            "models/lora.pt": "c4e43b87d62d4b0614b496773639f1ab170a7ee486dc23407901e9d3a5ebc07a",
        }
        for name, value in expected.items():
            with (dest / name).open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            if actual != value:
                raise ValueError(f"Official SHA256 mismatch: {name}")
        (dest / "official_sha256_verified.json").write_text(json.dumps(expected, indent=2))
        source = "modelscope:Efficient-Large-Model/LongLive-1.3B; SHA256 matches official Hugging Face"
    else:
        from huggingface_hub import snapshot_download
        dest = args.root / "models/LongLive-1.3B"
        snapshot_download("Efficient-Large-Model/LongLive-1.3B", local_dir=str(dest), max_workers=4)
        source = "huggingface:Efficient-Large-Model/LongLive-1.3B"
    files = [{"path": str(f.relative_to(dest)), "bytes": f.stat().st_size}
             for f in sorted(dest.rglob("*")) if f.is_file() and ".cache" not in f.parts]
    (dest / "download_inventory.json").write_text(json.dumps({"source": source, "files": files}, indent=2))


if __name__ == "__main__":
    main()
