import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from restream.runtime import ROOT, read_config, load_pipeline
from eval_reanchor import write_video

p = argparse.ArgumentParser()
p.add_argument("--reviewed", action="store_true")
p.add_argument("--config", default=str(ROOT / "configs/restream_mvp.yaml"))
a = p.parse_args()
if not a.reviewed:
    p.error("Run inference only after reviewing the preparation")
c = read_config(a.config)
torch.manual_seed(c["seed"])
pipe = load_pipeline(c, torch.device("cuda"))
shape = (1, (c["data"]["frames"] - 1) // 4 + 1, 16, c["data"]["height"] // 8, c["data"]["width"] // 8)
with torch.no_grad():
    pixels = pipe.inference(torch.randn(shape, device="cuda", dtype=torch.bfloat16),
                            ["A person walks slowly through a room, a realistic continuous shot."])
if not torch.isfinite(pixels).all():
    raise RuntimeError("Nonfinite baseline output")
out = ROOT / "outputs/baseline"
out.mkdir(parents=True, exist_ok=True)
write_video(out / "smoke.mp4", (pixels[0] * 255).byte().permute(0, 2, 3, 1).cpu().numpy(),
            float(c["data"]["fps"]), ["Official LongLive baseline"], -100)
