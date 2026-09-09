"""Content-addressed cache: video content + sample time + encoder/preprocessing identity."""
from pathlib import Path
import torch
from .reality_data import canonical_hash


class FeatureCache:
    def __init__(self, root, identity, tokens, channels):
        self.root, self.identity = Path(root), identity
        self.tokens, self.channels = tokens, channels

    def key(self, reference):
        return canonical_hash({"video_sha256": reference["video_sha256"], "time": reference["time"],
                               "encoder": self.identity})

    def path(self, reference):
        key = self.key(reference)
        return self.root / key[:2] / (key + ".pt")

    def write(self, reference, features):
        self.validate(features)
        path = self.path(reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save({"key": self.key(reference), "features": features.detach().cpu().float()}, temporary)
        temporary.replace(path)

    def validate(self, features):
        if features.shape != (self.tokens, self.channels) or not torch.isfinite(features).all():
            raise ValueError("Malformed/nonfinite cached visual features")

    def read(self, reference):
        path = self.path(reference)
        if not path.is_file():
            raise FileNotFoundError(f"Missing reference features; run scripts/cache_reality_features.py: {path}")
        record = torch.load(path, map_location="cpu", weights_only=True)
        if record["key"] != self.key(reference):
            raise ValueError("Feature cache identity mismatch")
        self.validate(record["features"])
        return record["features"].float()
