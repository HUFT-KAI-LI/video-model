"""Freeze source-level splits before feature extraction or optimizer construction."""
import argparse
import hashlib
import itertools
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from .io_utils import read_jsonl, sha256, write_json, write_jsonl


def caption_key(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def perceptual_signature(path):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 17:
            raise ValueError(f'Video has fewer than 17 frames: {path}')
        hashes = []
        for index in np.linspace(0, count - 1, 8).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {index}: {path}')
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32))
            dct = cv2.dct(gray.astype(np.float32))[:8, :8].flatten()[1:]
            bits = dct > np.median(dct)
            hashes.append(f'{sum(int(v) << i for i, v in enumerate(bits)):016x}')
        return hashes
    finally:
        cap.release()


def assign_groups(rows, seed=42):
    """Union all duplicate identities, preserving any previous training exposure."""
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i, j):
        parents[root(j)] = root(i)

    seen = {}
    for i, row in enumerate(rows):
        for field in ('source_id', 'sha256', 'caption_group', 'video'):
            key = (field, row[field])
            if key in seen:
                union(i, seen[key])
            seen[key] = i
    near_pairs = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        a, b = rows[i].get('perceptual_hashes'), rows[j].get('perceptual_hashes')
        if a and b and max((int(x, 16) ^ int(y, 16)).bit_count() for x, y in zip(a, b)) <= 6:
            union(i, j)
            near_pairs.append([rows[i]['source_id'], rows[j]['source_id']])
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    heldout = []
    for group in groups.values():
        group_id = hashlib.sha256('|'.join(sorted({r['sha256'] for r in group})).encode()).hexdigest()[:20]
        for row in group:
            row['group_id'] = group_id
        if any(r['previous_split'] == 'train' for r in group):
            for row in group:
                row['split'] = 'train'
        else:
            heldout.append(group)
    if len(heldout) < 2:
        raise ValueError('Need at least two non-training groups for separate validation and test sets')
    heldout.sort(key=lambda g: g[0]['group_id'])
    random.Random(seed).shuffle(heldout)
    boundary = max(1, len(heldout) // 2)
    for index, group in enumerate(heldout):
        for row in group:
            row['split'] = 'val' if index < boundary else 'test'
    return rows, near_pairs


def audit_splits(splits):
    for name, rows in splits.items():
        if not rows or any(r['split'] != name for r in rows):
            raise ValueError(f'Empty or mislabeled split: {name}')
    for a, b in itertools.combinations(splits, 2):
        for field in ('source_id', 'sha256', 'caption_group', 'group_id', 'video'):
            overlap = {r[field] for r in splits[a]} & {r[field] for r in splits[b]}
            if overlap:
                raise ValueError(f'Leakage between {a} and {b}: {field}={overlap}')
    return {name: len(rows) for name, rows in splits.items()}


def load_split(directory, split, training=False):
    directory = Path(directory)
    lock = json.loads((directory / 'split_lock.json').read_text())
    if training and split != 'train':
        raise ValueError('Optimizer inputs must come exclusively from train')
    splits = {}
    for name in ('train', 'val', 'test'):
        path = directory / f'{name}.jsonl'
        if sha256(path) != lock['manifest_sha256'][name]:
            raise ValueError(f'Frozen manifest modified: {path}')
        splits[name] = read_jsonl(path)
    audit_splits(splits)
    return splits[split]


def prepare(source, output, seed):
    source, output = Path(source), Path(output)
    if (output / 'split_lock.json').exists():
        raise FileExistsError('Split is already frozen; use a new output directory')
    rows = []
    for previous in ('train', 'val'):
        for item in read_jsonl(source / f'{previous}.jsonl'):
            path = Path(item['video']).resolve(strict=True)
            digest = sha256(path)
            if digest != item['sha256']:
                raise ValueError(f'Input video checksum mismatch: {path}')
            caption = item['caption'].strip()
            if not caption_key(caption):
                raise ValueError('Empty caption')
            rows.append(dict(video=str(path), caption=caption, source_id=item['source_id'],
                             sha256=digest, previous_split=previous, duration=item['duration'],
                             window_start=item['window_start'], caption_group=caption_key(caption),
                             perceptual_hashes=perceptual_signature(path), visual_review='pending'))
    rows, near = assign_groups(rows, seed)
    splits = {name: sorted([r for r in rows if r['split'] == name], key=lambda r: r['source_id'])
              for name in ('train', 'val', 'test')}
    counts = audit_splits(splits)
    for name, items in splits.items():
        for row in items:
            row['sample_id'] = hashlib.sha256((row['sha256'] + '|' + row['caption_group'] + '|' +
                                               str(row['window_start'])).encode()).hexdigest()[:24]
        write_jsonl(output / f'{name}.jsonl', items)
    lock = dict(seed=seed, counts=counts, near_duplicate_pairs=near,
                manifest_sha256={n: sha256(output / f'{n}.jsonl') for n in splits},
                source_manifest_sha256={n: sha256(source / f'{n}.jsonl') for n in ('train', 'val')},
                policy='Previous train stays train; remaining duplicate groups split val/test 1:1',
                limitations=['Source IDs do not identify shared shows/episodes.',
                             'Eight-frame pHash detects aligned near duplicates, not all edited copies.',
                             'Normalized captions are grouped; semantic paraphrases require human review.',
                             'Previous validation exposure means this is not a pristine external benchmark.',
                             'Clip captions and visual quality have not been manually verified.'])
    write_json(output / 'split_lock.json', lock)
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(args.source, args.output, args.seed)
