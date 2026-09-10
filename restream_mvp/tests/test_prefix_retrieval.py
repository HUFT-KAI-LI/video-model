import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_prefix_reference_retrieval import (derangements, retrieval_gate,
                                                     score_bank, similarity_columns)
from restream.reality_selection import select_targets


class PrefixRetrievalTests(unittest.TestCase):
    def test_seeded_seven_of_83_runs_complete_null_matrices(self):
        rows = [{'source_id': str(i)} for i in range(83)]
        indices = select_targets(rows, 7, 17)
        self.assertTrue(max(indices) >= 7)
        rng = torch.Generator().manual_seed(42)
        queries = torch.randn(7, 12, generator=rng)
        banks = {i: {key: torch.randn(2, 12, generator=rng) for key in
                     ('correct', 'easy_wrong', 'hard_wrong')} for i in indices}
        columns = {key: similarity_columns(queries, banks, key, indices) for key in banks[indices[0]]}
        permutations = derangements([rows[i]['source_id'] for i in indices], 1000, 17)
        self.assertEqual(len(permutations), 1000)
        for key, matrix in columns.items():
            self.assertEqual(matrix.shape, (7, 7))
            for j in range(7):
                for col, global_id in enumerate(indices):
                    self.assertAlmostEqual(matrix[j, col].item(), score_bank(queries[j], banks[global_id][key]), places=6)
        for permutation in permutations:
            for wrong in ('easy_wrong', 'hard_wrong'):
                actual = sum(columns['correct'][q, col] > columns[wrong][q, col] for col, q in enumerate(permutation))
                expected = sum(score_bank(queries[q], banks[i]['correct']) > score_bank(queries[q], banks[i][wrong])
                               for i, q in zip(indices, permutation))
                self.assertEqual(int(actual), expected)

    def test_gate_requires_both_easy_and_hard(self):
        good = {'accuracy_easy': 1., 'accuracy_hard': .94,
                'margin_easy': {'mean': .5, 'low': .4}, 'margin_hard': {'mean': .27, 'low': .23}}
        self.assertTrue(retrieval_gate(good)['pass'])
        for kind in ('easy', 'hard'):
            for value in (.52, .70, None, float('nan')):
                self.assertFalse(retrieval_gate({**good, f'accuracy_{kind}': value})['pass'])
            for margin in ({'mean': .1, 'low': 0}, {'mean': .1, 'low': -.1},
                           {'mean': None, 'low': None}, {'mean': -.1, 'low': .1}):
                self.assertFalse(retrieval_gate({**good, f'margin_{kind}': margin})['pass'])


if __name__ == '__main__':
    unittest.main()
