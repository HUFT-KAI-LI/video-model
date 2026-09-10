"""Reserve 100 unexposed sources; current 83 validation targets become R1 Dev.

No video decode or model scores. Previously rejected sources can be reserved,
so this freezes source membership, not a claim of 100 eligible test targets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import read_manifest
from restream.reality_splits import diagnostic_source_ids, digest, freeze_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-sources', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260910)
    args = parser.parse_args()
    inputs = {name: ROOT / 'data' / (name + '.jsonl') for name in
              ('train', 'val', 'reality_train', 'reality_val', 'reality_val_online')}
    rows = {name: read_manifest(path) for name, path in inputs.items()}
    # Only reports predating R1 participate; generated R1 checks cannot change
    # the frozen membership on a deterministic replay.
    lock_path = ROOT/'data/r1_split_lock.json'
    if lock_path.exists():
        prior = json.loads(lock_path.read_text())['prior_diagnostic_reports']
        reports = [ROOT/name for name in sorted(prior)]
        if any(digest(path) != prior[str(path.relative_to(ROOT))] for path in reports):
            raise ValueError('Historical diagnostic report changed after the source lock')
    else:
        reports = sorted(p for p in (ROOT / 'validation').rglob('*.json')
                         if 'r1_top1' not in p.parts)
    diagnostic_sources = diagnostic_source_ids(rows['train'] + rows['val'],
                                               [json.loads(p.read_text()) for p in reports])
    train, dev, test = freeze_sources(rows['train'], rows['val'],
                                     rows['reality_train'] + rows['reality_val'],
                                     rows['reality_val_online'], args.test_sources, args.seed, diagnostic_sources)
    outputs = {}
    for name, items in (('r1_train_sources', train), ('r1_dev_sources', dev), ('r1_test_sources', test)):
        outputs[f'data/{name}.jsonl'] = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in items)
        outputs[f'data/{name}.txt'] = ''.join(r['source_id'] + '\n' for r in items)
    lock = {
        'schema': 1, 'seed': args.seed, 'counts': {'train': len(train), 'dev': len(dev), 'test_reserved': len(test)},
        'source_manifests': {'train': 'data/r1_train_sources.jsonl', 'val': 'data/r1_dev_sources.jsonl'},
        'inputs': {str(p.relative_to(ROOT)): digest(p) for p in inputs.values()},
        'prior_diagnostic_reports': {str(p.relative_to(ROOT)): digest(p) for p in reports},
        'diagnostic_source_count': len(diagnostic_sources),
        'files': {name: hashlib.sha256(content.encode()).hexdigest() for name, content in outputs.items()},
        'test_status': 'source membership reserved; eligibility has not been evaluated',
        'policy': 'Exclude every old Reality Train/Val source and video SHA, including contributors to global means, '
                  'and explicit sources/video paths/SHAs from prior ReAnchor, VAE and Reality diagnostic reports. '
                  'Current 83 val targets are Dev. Test is forbidden for architecture, temperature, top-k, loss tuning or smoke tests. '
                  'Source quality filtering predates this split; reserved sources may fail fixed test eligibility. '
                  'Do not replace rejected test sources using model results.',
    }
    outputs['data/r1_split_lock.json'] = json.dumps(lock, indent=2, ensure_ascii=False) + '\n'
    # Idempotent replay is allowed; changing any existing frozen artifact fails.
    for name, content in outputs.items():
        path = ROOT / name
        if path.exists() and path.read_text() != content:
            raise FileExistsError(f'Refusing to change frozen split: {path}')
    for name, content in outputs.items():
        (ROOT / name).write_text(content)
    print(json.dumps(lock['counts']))


if __name__ == '__main__':
    main()
