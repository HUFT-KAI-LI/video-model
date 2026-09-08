import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from restream.anchor_adapter import GatedAnchorAdapter
from restream.dataset import VideoDataset
from restream.metrics import future_errors
from restream.objective import prepare
from restream.runtime import ROOT, read_config, load_pipeline, rollout


def write_video(path, frames, fps, labels, anchor_frame):
    from fractions import Fraction
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        for i, pixels in enumerate(frames):
            picture = Image.fromarray(pixels)
            draw = ImageDraw.Draw(picture)
            for j, label in enumerate(labels):
                x = j * picture.width // len(labels)
                draw.rectangle((x, 0, x + 190, 32), fill="black")
                draw.text((x + 4, 4), label, fill="white")
                if anchor_frame <= i <= anchor_frame + max(1, round(fps / 2)):
                    draw.text((x + 4, 18), "ANCHOR ARRIVES HERE", fill="yellow")
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(picture), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@torch.no_grad()
def evaluate(pipeline, adapter, config, device, output, cases=None, lpips_model=None):
    output.mkdir(parents=True, exist_ok=True)
    dataset = VideoDataset(ROOT / config["data"]["val_manifest"], config["data"]["frames"],
                           config["data"]["height"], config["data"]["width"])
    loader = DataLoader(dataset, batch_size=1)
    results = []
    for case, batch in enumerate(loader):
        if case >= (cases or config["eval"]["cases"]):
            break
        seed = config["seed"] + case
        gt, history, real, cond, anchor, arrival = prepare(pipeline, batch, device, config,
                                                         torch.Generator(device=device).manual_seed(seed), force_drift=True)
        noise = torch.randn(gt[:, anchor + 1:].shape, dtype=gt.dtype, device=device,
                            generator=torch.Generator(device=device).manual_seed(seed + 10000))
        variants = {"no_anchor": history, "hard_anchor": torch.cat((history[:, :-1], real), dim=1)}
        if adapter is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                corrected = adapter(history[:, -1:], real)
            variants["learned_anchor"] = torch.cat((history[:, :-1], corrected.to(history.dtype)), dim=1)
        folder = output / f"case_{case:03d}"
        folder.mkdir(exist_ok=True)
        pixels_gt = batch["pixels"][0].permute(1, 0, 2, 3).to(device)
        images = {"GT": pixels_gt}
        record = {"source_id": batch["source_id"][0], "seed": seed, "anchor_latent_index": anchor,
                  "anchor_pixel_index": 4 * anchor, "actual_anchor_sec": arrival,
                  "requested_anchor_sec": float(batch["anchor_sec"][0]), "variants": {}}
        window = float(batch["window_sec"][0])
        for name, prefix in variants.items():
            generated = rollout(pipeline, prefix, cond, noise.clone(),
                                torch.Generator(device=device).manual_seed(seed + 20000))
            if not torch.isfinite(generated).all():
                raise RuntimeError(f"Nonfinite rollout: {name}")
            scores = future_errors(generated, gt, anchor, window)
            decoded = pipeline.vae.decode_to_pixel(generated)[0]
            images[name] = decoded
            scores["future_lpips"] = None
            if lpips_model is not None:
                start = 4 * anchor + 1
                values = [lpips_model(decoded[i:i + 1].float(), pixels_gt[i:i + 1].float()).mean().item()
                          for i in range(start, decoded.shape[0])]
                scores["future_lpips"] = sum(values) / len(values)
            record["variants"][name] = scores
        arrays = {name: ((video.float().clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
                  for name, video in images.items()}
        fps = (config["data"]["frames"] - 1) / window
        for name, array in arrays.items():
            write_video(folder / f"{name}.mp4", array, fps, [name], 4 * anchor)
        write_video(folder / "comparison.mp4", np.concatenate(list(arrays.values()), axis=2), fps,
                    list(arrays), 4 * anchor)
        (folder / "metrics.json").write_text(json.dumps(record, indent=2))
        results.append(record)
    summary = {"cases": results, "lpips_status": "computed" if lpips_model is not None else "not requested", "aggregate": {}}
    for name in ("no_anchor", "hard_anchor", "learned_anchor"):
        entries = [r["variants"][name] for r in results if name in r["variants"]]
        if entries:
            summary["aggregate"][f"future_latent_mse_{name}"] = sum(e["future_latent_mse"] for e in entries) / len(entries)
            summary["aggregate"][f"future_lpips_{name}"] = (sum(e["future_lpips"] for e in entries) / len(entries)) if lpips_model else None
            if name != "no_anchor":
                summary["aggregate"][f"recovery_win_rate_{name}"] = sum(r["variants"][name]["future_latent_mse"] < r["variants"]["no_anchor"]["future_latent_mse"] for r in results) / len(results)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs/restream_mvp.yaml"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation")
    p.add_argument("--cases", type=int)
    p.add_argument("--lpips", action="store_true")
    p.add_argument("--reviewed", action="store_true")
    a = p.parse_args()
    if not a.reviewed:
        p.error("Evaluation is a post-review experiment; pass --reviewed after review")
    config = read_config(a.config)
    device = torch.device("cuda")
    pipeline = load_pipeline(config, device)
    adapter = None
    if a.checkpoint:
        path = a.checkpoint / "state.pt" if a.checkpoint.is_dir() else a.checkpoint
        state = torch.load(path, map_location="cpu", weights_only=False)
        adapter = GatedAnchorAdapter(config["reanchor"]["channels"]).to(device)
        adapter.load_state_dict(state["adapter"])
        adapter.eval()
    perceptual = None
    if a.lpips:
        import lpips
        perceptual = lpips.LPIPS(net="alex").eval().requires_grad_(False).to(device)
    evaluate(pipeline, adapter, config, device, a.output, a.cases, perceptual)


if __name__ == "__main__":
    main()
