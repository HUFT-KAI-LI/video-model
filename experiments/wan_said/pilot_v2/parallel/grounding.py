"""Parallel frozen encoders, preserving the original global contrastive batch and optimizer."""
import argparse
import copy
import queue
import traceback

import torch
import torch.multiprocessing as mp

from ...runtime import seed_everything
from ..config import load_config
from ..execution import deterministic_attention
from ..grounding import train
from ..online import OnlineFeatures


def shard_indices(count, workers):
    if count < 1 or workers < 1:
        raise ValueError('Positive sample/worker counts required')
    return [list(range(rank, count, workers)) for rank in range(workers)]


def worker(rank, cfg, incoming, outgoing):
    try:
        torch.cuda.set_device(rank)
        seed_everything(cfg['seed'])
        device = f'cuda:{rank}'
        cfg = copy.deepcopy(cfg)
        cfg['devices'] = {key: device for key in cfg['devices']}
        provider = OnlineFeatures(cfg)
        with deterministic_attention(cfg['deterministic_attention']):
            while True:
                request = incoming.get()
                if request is None:
                    break
                result = []
                for index, row in request:
                    features = provider(row, ('text', 'teacher'))
                    result.append((index, features['text']['pooled'], features['teacher']['pooled']))
                outgoing.put((rank, result, None))
    except BaseException:
        outgoing.put((rank, None, traceback.format_exc()))


class ParallelFeatures:
    def __init__(self, cfg, workers=4):
        if workers != 4 or torch.cuda.device_count() < workers:
            raise ValueError('This execution protocol requires four visible CUDA devices')
        ctx = mp.get_context('spawn')
        self.output = ctx.Queue()
        self.inputs = [ctx.Queue() for _ in range(workers)]
        self.processes = [ctx.Process(target=worker, args=(r, cfg, self.inputs[r], self.output))
                          for r in range(workers)]
        for process in self.processes:
            process.start()

    def encode_batch(self, rows, device):
        for channel, indices in zip(self.inputs, shard_indices(len(rows), len(self.inputs))):
            channel.put([(index, rows[index]) for index in indices])
        results, ranks = [], set()
        while len(ranks) < len(self.inputs):
            try:
                rank, values, error = self.output.get(timeout=10)
            except queue.Empty:
                if any(not p.is_alive() for p in self.processes):
                    raise RuntimeError('Frozen encoder worker exited before returning its features')
                continue
            if error or rank in ranks:
                raise RuntimeError(error or 'Duplicate worker response')
            ranks.add(rank)
            results.extend(values)
        results.sort(key=lambda item: item[0])
        if [v[0] for v in results] != list(range(len(rows))):
            raise RuntimeError('Parallel encoding dropped or duplicated samples')
        return (torch.cat([v[1] for v in results]).to(device),
                torch.cat([v[2] for v in results]).to(device))

    def close(self):
        for channel in self.inputs:
            channel.put(None)
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--val-limit', type=int, default=64)
    args = parser.parse_args()
    cfg = load_config(args.config)
    provider = ParallelFeatures(cfg)
    try:
        train(cfg, 'said', args.output, args.steps, val_limit=args.val_limit,
              resume=args.resume, provider=provider)
    finally:
        provider.close()
