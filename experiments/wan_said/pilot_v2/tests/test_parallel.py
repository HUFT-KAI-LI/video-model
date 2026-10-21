"""Execution-adapter tests. Synthetic gradients below are not experimental scores."""
import queue
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from experiments.wan_said.pilot_v2.parallel.grounding import ParallelFeatures, shard_indices
from experiments.wan_said.pilot_v2.parallel.train import assignment
from experiments.wan_said.pilot_v2.train import sample_index


def tail_worker(rank, rendezvous, result):
    dist.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=4)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.25)
        wrapped = DistributedDataParallel(model)
        position, scale = assignment(0, 3, 4, rank)
        x = torch.tensor([[float(position+1)]])
        loss = wrapped(x).square().mean()*scale
        loss.backward()
        if rank == 0:
            torch.save(model.weight.grad, result)
    finally:
        dist.destroy_process_group()


class ParallelTests(unittest.TestCase):
    def test_one_epoch_exact_3995_ids_without_sampler_padding_duplicates(self):
        positions = []
        for step in range(999):
            for rank in range(4):
                position, scale = assignment(step, 3995, 4, rank)
                if scale:
                    positions.append(position)
        self.assertEqual(positions, list(range(3995)))
        indices = [sample_index(p, 3995, 42) for p in positions]
        self.assertEqual(len(set(indices)), 3995)
        self.assertEqual(assignment(998, 3995, 4, 3)[1], 0)
        self.assertAlmostEqual(assignment(998, 3995, 4, 0)[1], 4/3)

    def test_encoder_shards_preserve_all_indices_including_short_batches(self):
        for count in (1, 3, 8, 64):
            shards = shard_indices(count, 4)
            self.assertEqual(sorted(i for shard in shards for i in shard), list(range(count)))
            self.assertLessEqual(max(map(len, shards))-min(map(len, shards)), 1)

    def test_encoder_results_reordered_before_global_contrastive_loss(self):
        provider = object.__new__(ParallelFeatures)
        provider.inputs = [queue.Queue() for _ in range(4)]
        provider.output = queue.Queue()
        for rank in (3, 1, 0, 2):
            provider.output.put((rank, [(i, torch.tensor([[float(i)]]), torch.tensor([[float(i+10)]]))
                                       for i in range(rank, 8, 4)], None))
        text, video = provider.encode_batch([dict(sample_id=str(i)) for i in range(8)], 'cpu')
        self.assertEqual(text.flatten().tolist(), list(range(8)))
        self.assertEqual(video.flatten().tolist(), list(range(10, 18)))

    def test_encoder_failure_not_silently_dropped(self):
        provider = object.__new__(ParallelFeatures)
        provider.inputs = [queue.Queue() for _ in range(4)]
        provider.output = queue.Queue()
        provider.output.put((2, None, 'fixture encoder failed'))
        with self.assertRaisesRegex(RuntimeError, 'fixture encoder failed'):
            provider.encode_batch([dict(sample_id='x')], 'cpu')

    def test_real_four_process_ddp_tail_equals_three_sample_mean_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = str(Path(tmp)/'gradient.pt')
            mp.spawn(tail_worker, args=(str(Path(tmp)/'rendezvous'), result), nprocs=4, join=True)
            actual = torch.load(result, weights_only=True)
        weight = torch.tensor(0.25, requires_grad=True)
        (weight*torch.arange(1., 4.)).square().mean().backward()
        self.assertTrue(torch.allclose(actual, weight.grad.reshape(1, 1)))


if __name__ == '__main__':
    unittest.main()
