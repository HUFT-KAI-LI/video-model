import os
from pathlib import Path
from modelscope import MsDataset

root = Path(__file__).resolve().parents[1]
os.environ["MODELSCOPE_CACHE"] = str(root / "data/sdk_cache")
ds = MsDataset.load("Youku-AliceMind", namespace="modelscope", subset_name="caption",
                    split="train", use_streaming=True, cache_dir=str(root / "data/sdk_cache"))
print("DATASET", type(ds), flush=True)
print("FIRST", next(iter(ds)), flush=True)
