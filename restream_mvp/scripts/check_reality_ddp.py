"""Two CPU/Gloo ranks, batch=1 each: no-memory rank + correct-memory rank, no optimizer."""
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.reality_memory import RealityMemory, memory_regularization
from restream.reality_data import write_json


def worker(rank, folder):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    dist.init_process_group("gloo", init_method="file://" + folder + "/rendezvous", rank=rank, world_size=2, timeout=timedelta(seconds=60))
    memory = RealityMemory(6, 8, 12, 2)
    model = DDP(memory)
    norms = []
    # Exercise two reducer iterations, including an entirely empty global batch.
    for iteration in range(2):
        model.zero_grad(set_to_none=True)
        context, features = torch.randn(1, 5, 12), torch.randn(1, 2, 3, 6)
        mask = torch.full((1, 2), rank == 1 and iteration == 0, dtype=torch.bool)
        fused, stats = model(context, features, mask)
        regularization, _ = memory_regularization(stats, torch.tensor([rank == 0]), 1e-5, .01)
        (fused.square().mean() + regularization).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
        norm = sum(p.grad.square().sum() for p in memory.parameters()).sqrt()
        norms.append(norm.item())
        peers = [torch.zeros_like(norm) for _ in range(2)]
        dist.all_gather(peers, norm)
        assert torch.equal(peers[0], peers[1])
        assert (norm.item() > 0) if iteration == 0 else (norm.item() == 0)
    if rank == 0:
        write_json(Path(folder) / "result.json", {"status": "passed", "world_size": 2, "backend": "gloo",
                   "batch_per_rank": 1, "gradient_norms": norms, "optimizer_steps": 0,
                   "scenarios": ["rank0 no-memory; rank1 correct-memory", "all ranks no-memory"]})
    dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="reality-ddp-") as folder:
        mp.spawn(worker, args=(folder,), nprocs=2, join=True)
        result = json.loads((Path(folder) / "result.json").read_text())
        write_json(ROOT / "validation/reality_memory/ddp_check.json", result)
        print(result)
