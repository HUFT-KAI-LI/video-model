"""Validate all saved frames and compute a clearly labeled whole-frame CLIP color proxy."""
import argparse
import json
from pathlib import Path
import sys
from protocol import ROOT, HERE, DESIGN, COLORS, MODELS, jobs, color_texts, digest, write_json, write_csv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    import av
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(4)
    run = args.run.resolve()
    assets = ROOT / 'restream_mvp/models/clip-vit-base-patch32'
    expected = 'a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
    if digest(assets / 'pytorch_model.bin') != expected:
        raise ValueError('CLIP weight checksum mismatch')
    model = CLIPModel.from_pretrained(assets, local_files_only=True).to(args.device).eval()
    processor = CLIPProcessor.from_pretrained(assets, local_files_only=True)
    output = run / 'evaluation'
    output.mkdir(exist_ok=True)
    write_json(output / 'provenance.json', dict(kind='AUTOMATIC_PROXY_NOT_HUMAN_ACCURACY',
        source_sha256={p.name: digest(p) for p in [HERE / 'evaluate.py', HERE / 'protocol.py']},
        clip_files={str(p.relative_to(assets)): digest(p) for p in sorted(assets.rglob('*')) if p.is_file()},
        design_sha256=digest(run / 'plan.json'), device=args.device, torch=str(torch.__version__),
        colors=COLORS, initial_frames=DESIGN['proxy_frames'], secondary_frames=DESIGN['secondary_proxy_frames'],
        decision='argmax of mean unscaled CLIP cosine across four initial frames; no threshold tuning',
        limitations='Whole-frame color association does not verify item presence, localization or fine-grained binding. Human review required.'))
    rows, validation = [], []
    for job in [j for m in MODELS for j in jobs(m)]:
        folder = run / 'outputs' / job['video_id']
        if not (folder / 'complete.json').exists():
            continue
        manifest = json.loads((folder / 'complete.json').read_text())
        if digest(folder / 'video.mp4') != manifest['video_sha256']:
            raise RuntimeError(f'Video checksum mismatch: {folder}')
        chosen, allpts = [], []
        indices = DESIGN['proxy_frames'] + DESIGN['secondary_proxy_frames']
        with av.open(str(folder / 'video.mp4')) as container:
            stream = container.streams.video[0]
            if (stream.width, stream.height) != (DESIGN['width'], DESIGN['height']):
                raise ValueError('Wrong resolution')
            for i, frame in enumerate(container.decode(stream)):
                allpts.append(float(frame.pts * frame.time_base))
                if i in indices:
                    chosen.append(frame.to_image())
        if len(allpts) != 80 or not np.allclose(allpts, np.arange(80)/16, atol=1e-6):
            raise ValueError(f'Frame count or PTS mismatch: {folder}')
        target, texts = color_texts(job)
        with torch.inference_mode():
            text_batch = processor(text=texts, return_tensors='pt', padding=True).to(args.device)
            txt = model.get_text_features(**text_batch)
            txt = torch.nn.functional.normalize(txt, dim=-1)
            ims = processor(images=chosen, return_tensors='pt').to(args.device)
            img = torch.nn.functional.normalize(model.get_image_features(**ims), dim=-1)
            similarity = (img @ txt.T).float().cpu().numpy()
        initial = similarity[:4].mean(axis=0)
        later = similarity[4:].mean(axis=0)
        target_id = COLORS.index(target)
        alternative = np.delete(initial, target_id).max()
        row = {k: job[k] for k in ['video_id', 'pair_id', 'model', 'prompt_id', 'seed', 'category', 'difficulty', 'motion_group']}
        row.update(target_color=target, predicted_color=COLORS[int(initial.argmax())],
                   initial_color_proxy_correct=int(initial.argmax() == target_id),
                   initial_margin=float(initial[target_id] - alternative),
                   later_predicted_color=COLORS[int(later.argmax())],
                   later_color_proxy_correct=int(later.argmax() == target_id),
                   generation_seconds=manifest['seconds'], peak_allocated_gib=manifest['peak_allocated_gib'],
                   peak_reserved_gib=manifest['peak_reserved_gib'], video_sha256=manifest['video_sha256'])
        rows.append(row)
        np.savez_compressed(output / f"{job['video_id']}.npz", similarity=similarity, frame_indices=indices,
                            colors=COLORS, texts=texts)
        sheet = Image.new('RGB', (416 * 4, 260), 'white')
        draw = ImageDraw.Draw(sheet)
        for k, im in enumerate(chosen[:4]):
            sheet.paste(im.resize((416, 240)), (416*k, 20))
            draw.text((416*k+4, 3), f'frame {indices[k]} / {indices[k]/16:.2f}s', fill='black')
        sheet.save(output / f"{job['video_id']}_initial.jpg", quality=90)
        validation.append(dict(video_id=job['video_id'], frames=80, fps=16, duration_sec=5,
                               resolution=[480,832], pts_verified=True, sha256=manifest['video_sha256']))
        print(f"EVALUATED {job['video_id']} initial_color_proxy={row['predicted_color']}", flush=True)
    if rows:
        write_csv(output / 'proxy_scores.csv', rows)
    write_json(output / 'validation.json', dict(completed=len(validation), expected=160, videos=validation))


if __name__ == '__main__':
    main()
