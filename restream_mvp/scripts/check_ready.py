"""Asset integrity and data split checks; no model execution."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
checks = {}
model = root / "models/LongLive-1.3B"
record = model / "official_sha256_verified.json"
checks["longlive_official_hash_record"] = record.is_file()
if record.is_file():
    for name, expected in json.loads(record.read_text()).items():
        with (model / name).open("rb") as f:
            checks[name] = hashlib.file_digest(f, "sha256").hexdigest() == expected
wan = root / "models/Wan2.1-T2V-1.3B"
for name in ["Wan2.1_VAE.pth", "diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl/tokenizer.json"]:
    checks["wan/" + name] = (wan / name).is_file() and (wan / name).stat().st_size > 0
groups = {}
for split in ("train", "val"):
    path = root / f"data/{split}.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    groups[split] = {r["source_id"] for r in rows}
    checks[split + "_files_exist"] = bool(rows) and all(Path(r["video"]).is_file() for r in rows)
checks["minimum_train_count"] = len(groups["train"]) >= 300
checks["minimum_val_count"] = len(groups["val"]) >= 40
checks["no_source_leakage"] = not groups["train"] & groups["val"]
print(json.dumps(checks, indent=2))
(root / "logs/readiness.json").write_text(json.dumps(checks, indent=2))
if not all(checks.values()):
    raise SystemExit("Preparation incomplete; see failed checks")
