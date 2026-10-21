import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..data import audit_splits, load_split
from ..io_utils import sha256, write_json, write_jsonl
from .config import digest, load_config, preprocessing, signature


def prepare(cfg):
    parent = Path(cfg['parent_manifests'])
    output = Path(cfg['manifests'])
    parent_lock = sha256(parent/'split_lock.json')
    expected = dict(parent_split_lock_sha256=parent_lock, preprocessing_hash=signature(cfg))
    if output.exists():
        lock = json.loads((output/'split_lock.json').read_text())
        if any(lock.get(k) != v for k, v in expected.items()):
            raise ValueError('Existing pilot manifests have different provenance; use a new directory')
        for split in ('train', 'val', 'test'):
            load_split(output, split)
        return lock
    splits = {s: load_split(parent, s) for s in ('train', 'val', 'test')}

    def verify(row):
        if sha256(row['video']) != row['sha256']:
            raise ValueError(f'Source content changed: {row["video"]}')
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(verify, [r for rows in splits.values() for r in rows]))
    for split, rows in splits.items():
        for row in rows:
            row['parent_sample_id'] = row['sample_id']
            row.pop('window_start', None)
            row.update(caption_scope='full_source_video', temporal_sampling_policy=cfg['temporal_sampling_policy'],
                       preprocessing_hash=signature(cfg), caption_review='pending')
            row['sample_id'] = digest([row['sha256'], row['caption'], signature(cfg)])[:24]
            row['split'] = split
    counts = audit_splits(splits)
    output.mkdir(parents=True)
    for split, rows in splits.items():
        write_jsonl(output/f'{split}.jsonl', rows)
    lock = dict(expected, counts=counts, seed=cfg['seed'], preprocessing=preprocessing(cfg),
                manifest_sha256={s: sha256(output/f'{s}.jsonl') for s in splits},
                policy='Preserve parent train/val/test membership; replace local-window sampling only',
                dataset_scope='Existing size-capped 5K pilot, NOT the official VideoREPA 64K training set',
                full_dataset_ready=False, pilot_dataset_ready=True,
                test_usage='Sealed; never training, Grounding, negative pools, tuning, or preview audit',
                limitations=['Original pilot has a <=4MB file selection bias.',
                             'Uniform PTS endpoint coverage does not prove all action semantics are retained.',
                             'Shared center crop can remove caption-relevant content; human review pending.'])
    write_json(output/'split_lock.json', lock)
    return lock


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config)), indent=2))
