"""Seal eligibility of the existing Test reservation using unchanged data rules.

No DINO, generator, features or task metrics. The policy is written before any
Test decode; outputs are immutable on replay, including an empty target set.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_reality_manifest import candidate, attach_references
from restream.dataset import read_manifest, VideoDataset
from restream.reality_dataset import RealityDataset
from restream.reality_data import canonical_hash
from restream.reality_runtime import read_reality_config
from restream.reality_selection import selection_identity, selection_config_hash
from restream.reality_splits import digest


def immutable_write(path, content):
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'Refusing to change sealed artifact: {path}')
        return
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/reality_memory_r1_top1.yaml')
    args = parser.parse_args()
    config = read_reality_config(args.config)
    lock = json.loads((ROOT/config['data']['r1_split_lock']).read_text())
    for name, expected in lock['files'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Source lock changed')
    inputs = ['data/r1_split_lock.json', 'data/r1_test_sources.jsonl', 'data/reality_shots.jsonl']
    code = ['scripts/build_reality_manifest.py', 'scripts/seal_r1_test.py', 'restream/dataset.py',
            'restream/reality_dataset.py', 'restream/reality_data.py', 'restream/reality_temporal.py']
    policy = {'schema': 1, 'selection_identity': selection_identity(config),
              'selection_config_hash': selection_config_hash(config),
              'filter': config['reality_memory']['filter'],
              'input_sha256': {p: digest(ROOT/p) for p in inputs},
              'implementation_sha256': {p: digest(ROOT/p) for p in code},
              'rules': 'Same Train/Dev candidate builder, unchanged RNG/gap/pool/scene/shot thresholds; '
                       'validate full target decode and strict prefix invariant. No source replacement. '
                       'Unexpected runtime/provenance failures abort rather than silently reject. '
                       'Same-split donors require >=2 targets; retain a singleton with no wrong bank.',
              'models_run': False}
    policy_path = ROOT/'data/r1_test_eligibility_policy.json'
    immutable_write(policy_path, json.dumps(policy, indent=2, ensure_ascii=False)+'\n')
    sources = read_manifest(ROOT/'data/r1_test_sources.jsonl')
    shots = {r['source_id']: r for r in read_manifest(ROOT/'data/reality_shots.jsonl')}
    def inspect(row):
        shot = shots[row['source_id']]
        if shot['video_sha256'] != row['sha256'] or shot['filter_hash'] != canonical_hash(config['reality_memory']['filter']):
            raise ValueError('Stale shot metadata')
        if digest(Path(row['video'])) != row['sha256']:
            raise ValueError('Test video content SHA differs from reserved source')
        reasons = []
        target = None
        if shot['error']:
            reasons.append('shot_filter_error:' + str(shot['error']))
        else:
            target = candidate(row, shot['shots'], config, reasons)
        if target is not None:
            # Validate using exactly the production Dataset's decode and invariant.
            import tempfile
            data = config['data']
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'target.jsonl'
                path.write_text(json.dumps({**target, 'references': [], 'reference_kind': 'none'})+'\n')
                ds = RealityDataset(path, None, data['frames'], data['height'], data['width'], data['fps'],
                                    preflight=False, selection_protocol='strict_online',
                                    selection_config_hash=selection_config_hash(config),
                                    prefix_latents=config['reality_memory']['objective']['prefix_latents'],
                                    temporal_sampling='causal_previous')
                sample = VideoDataset.__getitem__(ds, 0)
                ds._assert_prefix_matches_visible_until(target, sample)
        return target, {'source_id': row['source_id'], 'eligible': target is not None,
                        'rejection_reasons': [] if target is not None else sorted(set(reasons)),
                        'attempt_reasons': dict(Counter(reasons))}
    with ThreadPoolExecutor(max_workers=config['reality_memory']['filter']['workers']) as pool:
        checked = list(pool.map(inspect, sources))
    targets = [row for row, _ in checked if row is not None]
    if len(targets) >= 2:
        attach_references(targets, config)
    elif targets:
        targets[0].update(references=targets[0]['reference_sets']['async'][:2], reference_kind='async',
                          sample_id=canonical_hash({'source': targets[0]['source_id'], 'start': targets[0]['target_start'], 'kind': 'async'})[:24])
    content = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in targets)
    manifest_path = ROOT/'data/r1_test_targets.jsonl'
    report = {'schema': 1, 'status': 'sealed', 'reserved_sources': len(sources), 'eligible_targets': len(targets),
              'rejected_sources': len(sources)-len(targets),
              'rejection_counts': dict(Counter(reason for _, record in checked for reason in record['rejection_reasons'])),
              'sources': [r for _, r in checked], 'selection_identity': selection_identity(config),
              'policy_sha256': digest(policy_path), 'manifest_sha256': hashlib.sha256(content.encode()).hexdigest(),
              'manifest': 'data/r1_test_targets.jsonl', 'models_run': False, 'source_replacements': 0,
              'limitation': 'Eligibility only. An empty or small Test does not support final quality claims; do not replace sources based on model outcomes.'}
    immutable_write(manifest_path, content)
    immutable_write(ROOT/'data/r1_test_eligibility.json', json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print({k: report[k] for k in ['reserved_sources','eligible_targets','rejected_sources','rejection_counts']}, flush=True)


if __name__ == '__main__':
    main()
