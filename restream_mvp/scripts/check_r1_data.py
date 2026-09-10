"""Decode every R1 Train/Dev target and verify strict-online runtime invariants.

No encoder, generator, optimizer or Test access; feature caches are not needed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.dataset import VideoDataset, read_manifest
from restream.reality_data import write_json
from restream.reality_dataset import RealityDataset
from restream.reality_runtime import read_reality_config
from restream.reality_selection import manifest_digest, selection_config_hash
from restream.reality_splits import validate_r1_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/reality_memory_r1_top1.yaml')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/reality_memory/r1_top1/data_audit.json')
    args = parser.parse_args()
    config = read_reality_config(args.config)
    data, memory = config['data'], config['reality_memory']
    if memory['stage'] != 'r1a':
        raise ValueError('Use the R1-A configuration with sealed Test sources')
    report = {'status': 'passed', 'optimizer_steps': 0, 'test_decoded': 0, 'splits': {},
              'split_lock_sha256': manifest_digest(ROOT / data['r1_split_lock'])}
    for split in ('train', 'val'):
        path = ROOT / data[f'{split}_manifest']
        rows = read_manifest(path)
        validate_r1_split(ROOT, config, split, rows)
        dataset = RealityDataset(path, None, data['frames'], data['height'], data['width'], data['fps'],
                                 preflight=False, selection_protocol='strict_online',
                                 selection_config_hash=selection_config_hash(config),
                                 prefix_latents=memory['objective']['prefix_latents'], temporal_sampling='causal_previous')
        def check(index):
            sample = VideoDataset.__getitem__(dataset, index)
            dataset._assert_prefix_matches_visible_until(dataset.rows[index], sample)
            return dataset.rows[index]['sample_id']
        checked = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for sample_id in pool.map(check, range(len(dataset))):
                checked.append(sample_id)
                if len(checked) % 100 == 0:
                    print(f'{split}: decoded {len(checked)}/{len(dataset)}', flush=True)
        report['splits'][split] = {'role': 'dev' if split == 'val' else 'train', 'targets': len(checked),
                                   'manifest_sha256': manifest_digest(path), 'sample_ids': checked}
    write_json(args.output, report)
    print({k: v['targets'] for k, v in report['splits'].items()}, flush=True)


if __name__ == '__main__':
    main()
