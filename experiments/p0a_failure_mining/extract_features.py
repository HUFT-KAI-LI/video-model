#!/usr/bin/env python3
"""DINOv2 appearance + CLIP attribute margins; whole-frame screening or supplied subject boxes."""
import argparse
import json
from pathlib import Path
from common import completed, digest, signature, write_csv, write_json


def sample_indices(blocks, count):
    import numpy as np
    if count < 1:
        raise ValueError('frames_per_block must be positive')
    return {b['block_index']: sorted(set(np.linspace(b['frame_start'], b['frame_end'] - 1,
                                                    count).round().astype(int).tolist())) for b in blocks}


def score_features(features, target_scores, alternate_scores, block_ids, reference_blocks):
    import numpy as np
    features = np.asarray(features, dtype=float)
    if not np.isfinite(features).all() or np.any(np.linalg.norm(features, axis=1) == 0):
        raise ValueError('Invalid DINO features')
    features = features / np.linalg.norm(features, axis=1, keepdims=True)
    mask = np.asarray(block_ids) < reference_blocks
    if not mask.any():
        raise ValueError('No early reference samples')
    reference = features[mask].mean(axis=0)
    norm = np.linalg.norm(reference)
    if norm < 1e-12:
        raise ValueError('Degenerate early reference')
    subject = features @ (reference / norm)
    margin = np.asarray(target_scores) - np.asarray(alternate_scores).max(axis=1)
    if not np.isfinite(margin).all():
        raise ValueError('Invalid CLIP scores')
    return subject, margin


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dino-model', type=Path, help='Offline local copy; all model/tokenizer files are hashed')
    p.add_argument('--clip-model', type=Path, help='Offline local copy; must match the planned model identity')
    p.add_argument('--boxes', type=Path, help='JSON: video_id -> frame_index -> [left,top,right,bottom] pixel coordinates; every sampled frame required')
    args = p.parse_args()
    import av
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    run = args.run_dir.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    cfg = dict(plan['config']['features'])
    local_hashes = {}
    for kind in ('dino', 'clip'):
        path = getattr(args, kind + '_model')
        if path is None:
            continue
        path = path.resolve()
        source = json.loads((path / 'provenance.json').read_text())
        if source['model'] != cfg[kind + '_model']:
            raise ValueError(f'{kind} local assets do not match the planned model identity')
        local_hashes[kind] = {str(f.relative_to(path)): digest(f) for f in sorted(path.rglob('*'))
                              if f.is_file() and '.cache' not in f.parts}
        cfg[kind + '_model'] = str(path)
    rows = completed(run)
    if not rows:
        raise ValueError('No completed real videos')
    boxes = json.loads(args.boxes.read_text()) if args.boxes else None
    if cfg['mode'] != 'whole_frame':
        raise ValueError('Config mode must be whole_frame; --boxes explicitly overrides it')
    dino_processor = AutoImageProcessor.from_pretrained(cfg['dino_model'], revision=cfg['dino_revision'])
    dino = AutoModel.from_pretrained(cfg['dino_model'], revision=cfg['dino_revision']).eval().to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(cfg['clip_model'], revision=cfg['clip_revision'])
    clip = CLIPModel.from_pretrained(cfg['clip_model'], revision=cfg['clip_revision']).eval().to(args.device)
    evidence = dict(plan_signature=plan['signature'], config=cfg, mode='subject_crop' if boxes else 'whole_frame',
                    local_asset_sha256=local_hashes,
                    boxes_sha256=digest(args.boxes) if args.boxes else None,
                    dino_commit=dino.config._commit_hash, clip_commit=clip.config._commit_hash,
                    extraction_source_sha256=digest(__file__), torch=str(torch.__version__),
                    transformers=__import__('transformers').__version__)
    cache = run / 'outputs/feature_cache'
    cache.mkdir(parents=True, exist_ok=True)
    previous = cache / 'provenance.json'
    if previous.exists() and json.loads(previous.read_text()) != evidence:
        raise ValueError('Feature provenance changed; use a separate run copy for a different feature protocol')
    write_json(previous, evidence)
    block_map = json.loads((run / 'block_map.json').read_text())['blocks']
    samples = sample_indices(block_map, cfg['frames_per_block'])
    wanted = {index: block for block, indices in samples.items() for index in indices}
    all_scores = []
    with torch.inference_mode():
        for row in rows:
            out = run / 'outputs' / row['video_id']
            video = out / 'video.mp4'
            if digest(video) != row['artifact_sha256']['video.mp4']:
                raise ValueError(f'Video changed: {video}')
            vectors, targets, alternatives, ids, indices, thumbnails = [], [], [], [], [], []
            text = [row['attribute_text']] + row['alternatives']
            text_inputs = clip_processor(text=text, return_tensors='pt', padding=True).to(args.device)
            text_features = clip.get_text_features(**text_inputs)
            text_features = torch.nn.functional.normalize(text_features, dim=-1)
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                if float(stream.average_rate) != row['fps']:
                    raise ValueError('Encoded FPS differs from manifest')
                count = 0
                for index, frame in enumerate(container.decode(video=0)):
                    count += 1
                    if index not in wanted:
                        continue
                    image = frame.to_image().convert('RGB')
                    if boxes is not None:
                        try:
                            x0, y0, x1, y1 = boxes[row['video_id']][str(index)]
                        except KeyError as error:
                            raise ValueError(f'Missing subject crop at {row["video_id"]}:{index}') from error
                        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                            raise ValueError('Invalid subject box')
                        image = image.crop((x0, y0, x1, y1))
                    dino_inputs = dino_processor(images=image, return_tensors='pt').to(args.device)
                    vectors.append(dino(**dino_inputs).last_hidden_state[:, 0].float().cpu().numpy()[0])
                    image_inputs = clip_processor(images=image, return_tensors='pt').to(args.device)
                    image_features = torch.nn.functional.normalize(clip.get_image_features(**image_inputs), dim=-1)
                    scores = (image_features @ text_features.T)[0].float().cpu().numpy()
                    targets.append(scores[0]); alternatives.append(scores[1:])
                    ids.append(wanted[index]); indices.append(index)
                    if index == samples[wanted[index]][0]:
                        image.thumbnail((160, 96))
                        thumbnails.append((wanted[index], image.copy()))
                if count != row['num_frames'] or len(indices) != len(wanted):
                    raise ValueError('Video is truncated or has unexpected frame count')
            subject, margin = score_features(vectors, targets, alternatives, ids,
                                            plan['config']['mining']['reference_blocks'])
            np.savez_compressed(cache / f"{row['video_id']}.npz", dino=np.array(vectors), target=targets,
                                alternatives=alternatives, frame_index=indices, block_index=ids,
                                subject=subject, attribute_margin=margin)
            for b in block_map:
                mask = np.array(ids) == b['block_index']
                all_scores.append(dict(video_id=row['video_id'], block_index=b['block_index'],
                                       start_sec=b['start_sec'], end_sec=b['end_sec'], n_frames=int(mask.sum()),
                                       subject=float(np.median(subject[mask])),
                                       attribute=float(np.median(margin[mask])), feature_mode=evidence['mode']))
            sheet = Image.new('RGB', (800, 120 * ((len(thumbnails) + 4)//5)), 'white')
            draw = ImageDraw.Draw(sheet)
            for j, (b, image) in enumerate(thumbnails):
                x, y = (j % 5)*160, (j//5)*120
                sheet.paste(image, (x, y))
                draw.text((x+3, y+97), f"block {b} / {block_map[b]['start_sec']:.2f}s", fill='black')
            sheet.save(out / 'preview.jpg')
            print('Scored:', row['video_id'], flush=True)
    write_csv(run / 'manifests/block_scores.csv', all_scores)
    write_json(run / 'manifests/block_scores.provenance.json', dict(feature_signature=signature(evidence),
               block_scores_sha256=digest(run / 'manifests/block_scores.csv'), videos=[r['video_id'] for r in rows]))


if __name__ == '__main__':
    main()
