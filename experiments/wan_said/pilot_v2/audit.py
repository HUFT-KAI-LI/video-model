import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..data import load_split
from ..io_utils import sha256, write_json
from ..runtime import upstream
from .config import load_config, signature
from .teacher import load_teacher
from .video import sample_video, teacher_input


def quantiles(values):
    return {name: float(np.quantile(values, q)) for name, q in
            [('p50', .5), ('p90', .9), ('p95', .95), ('p99', .99), ('max', 1)]}


def audit(cfg, output, count=8, teacher_check=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh audit output; do not overwrite human review')
    output.mkdir(parents=True)
    splits = {s: load_split(cfg['manifests'], s) for s in ('train', 'val', 'test')}
    all_rows = [r for rows in splits.values() for r in rows]
    upstream()
    from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer
    tokenizer = HuggingfaceTokenizer(str(Path(cfg['wan_weights'])/'google/umt5-xxl'),
                                     seq_len=512, clean='whitespace', local_files_only=True)
    text_lengths = {}
    for row in all_rows:
        ids = tokenizer(row['caption'], padding=False, truncation=False, add_special_tokens=True)
        text_lengths[row['sample_id']] = ids.shape[-1]
    # Do not preview or encode sealed test videos while developing the algorithm.
    candidates = []
    for split in ('train', 'val'):
        families = sorted({r['family'] for r in splits[split]})
        for family in families:
            family_rows = [r for r in splits[split] if r['family'] == family]
            candidates.extend(family_rows[:max(1, count//(2*len(families)))])
    candidates = candidates[:count]
    teacher = load_teacher(cfg, cfg['devices']['teacher']) if teacher_check else None
    report, review = [], []
    for row in candidates:
        images, sampling = sample_video(row, cfg)
        sheet = Image.new('RGB', (4*240, 2*180), 'white')
        draw = ImageDraw.Draw(sheet)
        for n, index in enumerate(np.linspace(0, cfg['frames']-1, 8).round().astype(int)):
            image = Image.fromarray((images[index].permute(1, 2, 0).numpy()*255).astype('uint8'))
            image.thumbnail((240, 150))
            x, y = (n % 4)*240, (n//4)*180
            sheet.paste(image, (x, y))
            draw.text((x+3, y+152), f'{sampling["wan_sampled_timestamps"][index]:.3f}s', fill='black')
        path = output/f'{row["sample_id"]}.jpg'
        sheet.save(path)
        item = dict(sample_id=row['sample_id'], split=row['split'], family=row['family'],
                    video=row['video'], preview=str(path.resolve()), sampling=sampling,
                    automatic_caption_alignment_proven=False, human_review='pending')
        item['caption_tokens_before_native_truncation'] = text_lengths[row['sample_id']]
        item['caption_truncated_at_native_512_limit'] = text_lengths[row['sample_id']] > 512
        if teacher is not None:
            results = []
            for frames in (17, 49, 81):
                local = dict(cfg, frames=frames)
                sampled, local_sampling = (images, sampling) if frames == cfg['frames'] else sample_video(row, local)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    inputs = teacher_input(sampled, local).to(cfg['devices']['teacher'])
                    first = teacher(inputs)
                    second = teacher(inputs)
                maximum = (first['raw_tokens']-second['raw_tokens']).abs().max().item()
                if maximum != 0 or first['raw_tokens'].requires_grad:
                    raise AssertionError('Frozen teacher is not deterministic/detached')
                results.append(dict(wan_frames=frames, shapes=teacher.last_shapes,
                                    sampled_directly_from_source=True,
                                    wan_sampled_timestamps=local_sampling['wan_sampled_timestamps'],
                                    teacher_sampled_timestamps=local_sampling['teacher_sampled_timestamps'],
                                    output_dim=first['pooled'].shape[-1], repeat_max_error=maximum,
                                    all_finite=bool(torch.isfinite(first['raw_tokens']).all()),
                                    teacher_has_grad=any(p.grad is not None for p in teacher.parameters())))
            item['teacher_checks'] = results
        report.append(item)
        review.append(dict(sample_id=row['sample_id'], split=row['split'], caption=row['caption'],
                           preview=str(path.resolve()), event_coverage='', caption_match='',
                           crop_preserves_subject='', approve='', rejection_reason='', reviewer=''))
        print('Audited', row['sample_id'], row['split'], flush=True)
    if review:
        with (output/'human_review.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(review[0]))
            writer.writeheader()
            writer.writerows(review)
    result = dict(scope='Existing 5K size-capped pilot, not official 64K',
                  split_counts={s: len(r) for s, r in splits.items()},
                  split_lock_sha256=sha256(Path(cfg['manifests'])/'split_lock.json'),
                  preprocessing_hash=signature(cfg),
                  duration_seconds=quantiles([r['duration'] for r in all_rows]),
                  source_fps=quantiles([r['actual_fps'] for r in all_rows]),
                  aspect_ratio=quantiles([r['width']/r['height'] for r in all_rows]),
                  caption_token_lengths=quantiles(list(text_lengths.values())),
                  native_text_token_limit=512,
                  captions_exceeding_native_text_limit=sum(n > 512 for n in text_lengths.values()),
                  text_limit_policy='Retain original caption and native tokenizer; flag truncation, do not silently rewrite captions',
                  family_counts=dict(Counter(r['family'] for r in all_rows)),
                  automated_review_count=len(report), samples=report,
                  teacher_weight_load=teacher.load_report if teacher else None,
                  human_review_completed=False, benchmark_efficacy_proven=False,
                  temporal_coverage_note='First/last decoded PTS coverage only, not proof of semantic action retention',
                  feature_cache_bytes=0, test_videos_previewed_or_encoded=False)
    write_json(output/'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=8)
    parser.add_argument('--teacher-check', action='store_true')
    args = parser.parse_args()
    audit(load_config(args.config), args.output, args.count, args.teacher_check)
