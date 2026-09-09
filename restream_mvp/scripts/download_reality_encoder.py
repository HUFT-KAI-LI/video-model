"""Download pinned DINOv2-S weights from ModelScope and verify official SHA-256."""
import hashlib
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3368852df502d7b60bf506eb3abde87533f55164"
FILES = {
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}


def main():
    destination = ROOT / "models/dinov2-small"
    destination.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
    for name, expected in FILES.items():
        path = destination / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
            continue
        url = f"https://modelscope.cn/models/facebook/dinov2-small/resolve/{REVISION}/{name}"
        partial = path.with_suffix(path.suffix + ".partial")
        with session.get(url, stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            digest, size = hashlib.sha256(), 0
            with partial.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    size += len(chunk)
                    if size > 100_000_000:
                        raise ValueError("Encoder download exceeded expected size budget")
                    digest.update(chunk)
                    output.write(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        partial.replace(path)
        print(f"Verified {name}: {size} bytes", flush=True)
    report = {"model": "facebook/dinov2-small", "source": "ModelScope", "revision": REVISION,
              "official_weight_reference": "https://huggingface.co/facebook/dinov2-small/blob/main/model.safetensors",
              "sha256": FILES, "weight_matches_official": True}
    (destination / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
