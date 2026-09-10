import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_r1 import R1CandidateBank, encode_prefix, select_r1_memory
from restream.reality_runtime import read_reality_config
from restream.reality_splits import diagnostic_source_ids, digest, freeze_sources, validate_r1_split


class R1InputsTests(unittest.TestCase):
    def test_prefix_encoder_never_reads_future_pixels(self):
        pixels = torch.linspace(-1, 1, 57)[None, None, :, None, None].expand(1, 3, 57, 4, 4).clone()
        encoder = Mock()
        encoder.encode_visual.side_effect = lambda images: images.mean((-1, -2))[:, :, None].expand(-1, -1, 8, -1)
        first, indices = encode_prefix(encoder, pixels, 6)
        self.assertEqual(indices, [0, 10, 20])
        self.assertEqual(first.shape, (1, 24, 3))
        pixels[:, :, 21:] = float('nan')
        second, _ = encode_prefix(encoder, pixels, 6)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        expected = ((pixels[:, :, indices].transpose(1, 2) + 1) / 2)
        torch.testing.assert_close(encoder.encode_visual.call_args.args[0], expected)

    def test_hard_bank_is_same_split_and_preserves_individual_tokens(self):
        features = [torch.tensor([1., 0.]).repeat(2, 8, 1),
                    torch.tensor([.99, .01]).repeat(2, 8, 1),
                    torch.tensor([.8, .2]).repeat(2, 8, 1),
                    torch.tensor([.7, .3]).repeat(2, 8, 1)]
        rows = [{'source_id': str(i), 'sha256': str(i), 'split': 'train' if i != 1 else 'val',
                 'reference_sets': {'async': [(i, 0), (i, 1)]}} for i in range(4)]
        ds = SimpleNamespace(rows=rows, reference_features=lambda refs: torch.stack([features[i][j] for i, j in refs]))
        bank = R1CandidateBank(ds, 2)
        candidates, refs, donor = bank[0]
        self.assertEqual(donor, 2)
        torch.testing.assert_close(candidates, torch.cat((features[0], features[2]))[None])
        prefix = features[0][0][None]
        routed, _, _ = select_r1_memory('routed', prefix, candidates)
        correct, _, _ = select_r1_memory('correct_only', prefix, candidates)
        global_features, _, _ = select_r1_memory('global_async', prefix, candidates, global_async=features[3][0])
        torch.testing.assert_close(routed[0, 0], features[0][0])
        torch.testing.assert_close(correct[0], features[0])
        torch.testing.assert_close(global_features[0], features[3])
        with self.assertRaises(ValueError):
            select_r1_memory('global_async', prefix, candidates)

    def test_r1_config_requires_top1_and_frozen_split(self):
        config = read_reality_config(ROOT / 'configs/reality_memory_r1_top1.yaml')
        import yaml
        for field, value in (('mode', 'soft'), ('reference_count', 0), ('prefix_frames', 22)):
            bad = copy.deepcopy(config)
            bad['reality_memory']['router'][field] = value
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'bad.yaml'
                path.write_text(yaml.safe_dump(bad))
                with self.assertRaises(ValueError):
                    read_reality_config(path)


class R1SplitTests(unittest.TestCase):
    def test_prior_diagnostics_exclude_video_paths_and_shas(self):
        rows = [{'source_id': str(i), 'sha256': f'hash{i}', 'video': f'/video/{i}', 'split': 'train'} for i in range(20)]
        reports = [{'cases': [{'source_id': '0'}, {'video': '/video/1'}, {'video_sha256': 'hash2'}]}]
        exposed = diagnostic_source_ids(rows, reports)
        self.assertEqual(exposed, {'0', '1', '2'})
        _, _, test = freeze_sources(rows, [], [], [], 17, 42, exposed)
        self.assertFalse({r['source_id'] for r in test} & exposed)

    def test_reserved_sources_exclude_old_training_dev_and_duplicate_content(self):
        train = [{'source_id': str(i), 'sha256': str(i), 'split': 'train'} for i in range(30)]
        val = [{'source_id': str(i), 'sha256': str(i), 'split': 'val'} for i in range(30, 40)]
        train[10]['sha256'] = train[0]['sha256']  # Same content under another source ID is exposed too.
        a, b, c = freeze_sources(train, val, train[:10] + val[:2], val[:2], 7, 42)
        self.assertEqual((a, b, c), freeze_sources(train, val, train[:10] + val[:2], val[:2], 7, 42))
        test_ids = {r['source_id'] for r in c}
        test_hashes = {r['sha256'] for r in c}
        self.assertEqual(len(c), 7)
        self.assertFalse(test_ids & {r['source_id'] for r in train[:11] + val[:2]})
        self.assertFalse(test_hashes & {r['sha256'] for r in a + b})
        self.assertEqual(b, val[:2])
        with self.assertRaises(ValueError):
            freeze_sources(train, val, train + val, val[:2], 7, 42)

    def test_locked_split_rejects_test_target_donor_and_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'train.jsonl'
            path.write_text(json.dumps({'source_id': 'train', 'sha256': 'trainhash'}) + '\n')
            lock = {'files': {'train.jsonl': digest(path)}, 'source_manifests': {'train': 'train.jsonl'}}
            (root / 'lock.json').write_text(json.dumps(lock))
            config = {'data': {'r1_split_lock': 'lock.json', 'source_train_manifest': 'train.jsonl'}}
            row = {'source_id': 'train', 'sha256': 'trainhash', 'split': 'train'}
            validate_r1_split(root, config, 'train', [row])
            with self.assertRaises(ValueError):
                validate_r1_split(root, config, 'test', [])
            with self.assertRaises(ValueError):
                validate_r1_split(root, config, 'train', [{**row, 'source_id': 'test'}])
            donor = {'source_id': 'test', 'video_sha256': 'testhash', 'split': 'train'}
            with self.assertRaises(ValueError):
                validate_r1_split(root, config, 'train', [{**row, 'reference_sets': {'wrong': [donor]}}])
            path.write_text(path.read_text() + '\n')
            with self.assertRaises(ValueError):
                validate_r1_split(root, config, 'train', [row])


if __name__ == '__main__':
    unittest.main()
