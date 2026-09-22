"""Regenerate and verify the complete local delta against the pinned source archive."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ("utils/wan_wrapper.py", "wan/modules/causal_model.py", "wan/modules/causal_model_infinity.py",
         "pipeline/causal_inference.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=ROOT / "code/longlive-v1.0.tar.gz")
    parser.add_argument("--write", action="store_true", help="Regenerate longlive.patch before verification")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    provenance = json.loads((ROOT / "code/source_provenance.json").read_text())
    with args.archive.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    if digest != provenance["archive_sha256"]:
        raise RuntimeError("Source archive SHA-256 does not match pinned provenance")
    with tarfile.open(args.archive) as archive:
        originals = {name: archive.extractfile("LongLive-1.0/" + name).read() for name in FILES}
    current = {name: (ROOT / "code/LongLive" / name).read_bytes() for name in FILES}
    patch = "".join("".join(difflib.unified_diff(originals[name].decode().splitlines(True),
                                               current[name].decode().splitlines(True),
                                               fromfile="a/" + name, tofile="b/" + name)) for name in FILES)
    patch_path = ROOT / "longlive.patch"
    if args.write:
        patch_path.write_text(patch)
    if patch_path.read_text() != patch:
        raise RuntimeError("longlive.patch is stale; regenerate with --write")
    with tempfile.TemporaryDirectory(prefix="restream-patch-") as folder:
        folder = Path(folder)
        for name, content in originals.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch_path)], cwd=folder, check=True)
        for name, content in current.items():
            if (folder / name).read_bytes() != content:
                raise RuntimeError(f"Reconstructed source differs: {name}")
    result = {"status": "passed", "archive_sha256": digest, "upstream_commit": provenance["commit"],
              "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
              "reconstructed_sha256": {name: hashlib.sha256(value).hexdigest() for name, value in current.items()}}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
