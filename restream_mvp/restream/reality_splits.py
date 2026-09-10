"""Freeze R1 source membership before model design, including donor isolation."""
import hashlib
import json
import random
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_sources(base_train, base_val, exposed, dev_rows, count=100, seed=20260910,
                   diagnostic_sources=()):
    all_rows = base_train + base_val
    by_source = {row['source_id']: row for row in all_rows}
    if len(by_source) != len(all_rows):
        raise ValueError('Base manifests must have unique source IDs')
    exposed_ids = {r['source_id'] for r in exposed + dev_rows} | set(diagnostic_sources)
    exposed_hashes = {r['sha256'] for r in exposed + dev_rows}
    exposed_hashes.update(r['sha256'] for r in all_rows if r['source_id'] in exposed_ids)
    eligible = sorted(r['source_id'] for r in all_rows
                      if r['source_id'] not in exposed_ids and r['sha256'] not in exposed_hashes)
    if type(count) is not int or count < 2 or len(eligible) < count:
        raise ValueError(f'Need {count} unexposed sources; only {len(eligible)} available')
    test_ids = sorted(random.Random(seed).sample(eligible, count))
    test_hashes = {by_source[s]['sha256'] for s in test_ids}
    dev_ids = {r['source_id'] for r in dev_rows}
    dev_hashes = {r['sha256'] for r in dev_rows}
    train = [r for r in base_train if r['source_id'] not in test_ids and
             r['source_id'] not in dev_ids and r['sha256'] not in test_hashes | dev_hashes]
    dev = [{**by_source[r['source_id']], 'split': 'val'} for r in dev_rows]
    test = [{**by_source[s], 'split': 'test'} for s in test_ids]
    if len({r['sha256'] for r in test}) != len(test):
        raise ValueError('Reserved test contains duplicate video content')
    return train, dev, test


def diagnostic_source_ids(base_rows, reports):
    """Match explicit source IDs, video paths and content SHAs in prior reports.

    Include older ReAnchor/VAE diagnostics, not only the Reality manifests.
    """
    lookup = {str(row[key]): row['source_id'] for row in base_rows
              for key in ('source_id', 'sha256', 'video') if key in row}
    found = set()
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str) and value in lookup:
            found.add(lookup[value])
    for report in reports:
        visit(report)
    return found


def validate_r1_split(root, config, split, rows):
    """Check frozen lists, target membership and every donor, before decode/cache."""
    lock_path = config['data'].get('r1_split_lock')
    if not lock_path:
        return
    if split not in ('train', 'val'):
        raise ValueError('R1 Test is sealed; only train/dev are available to development tools')
    root = Path(root)
    lock = json.loads((root / lock_path).read_text())
    for name, expected in lock['files'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Frozen R1 source split changed: {name}')
    source_file = config['data'][f'source_{split}_manifest']
    if source_file != lock['source_manifests'][split]:
        raise ValueError('R1 source manifest differs from the frozen split')
    allowed = {r['source_id']: r['sha256'] for r in
               map(json.loads, (root / source_file).read_text().splitlines())}
    for row in rows:
        if row['split'] != split or allowed.get(row['source_id']) != row['sha256']:
            raise ValueError('R1 target is outside its frozen source split')
        refs = list(row.get('references', []))
        for pool in row.get('reference_sets', {}).values():
            refs.extend(pool)
        if any(ref['split'] != split or allowed.get(ref['source_id']) != ref['video_sha256'] for ref in refs):
            raise ValueError('R1 reference donor is outside its frozen source split')
