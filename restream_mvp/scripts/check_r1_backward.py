"""One real LongLive R1 forward/backward with a fresh zero-init adapter.

No optimizer steps, resume, evaluation sweep or effect training. Optional
matched baseline inputs use the same adapter initialization and noise seed.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from restream.objective import future_loss
from restream.reality_data import write_json
from restream.reality_dataset import collate_reality
from restream.reality_encoder import RealityEncoder
from restream.reality_r1 import R1CandidateBank, encode_prefix, select_r1_memory
from restream.reality_runtime import (read_reality_config, make_memory, make_dataset,
                                      prepare_reality, PreserveHistory)
from restream.reality_selection import manifest_digest, selection_config_hash
from restream.runtime import load_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/reality_memory_r1_top1.yaml')
    parser.add_argument('--branch', choices=('routed', 'correct_only', 'global_async'), default='routed')
    parser.add_argument('--output', type=Path, default=ROOT / 'validation/reality_memory/r1_top1/real_backward.json')
    args = parser.parse_args()
    config = read_reality_config(args.config)
    if config['reality_memory']['stage'] != 'r1a':
        raise ValueError('Use the strict-online R1-A configuration')
    if not torch.cuda.is_available():
        raise RuntimeError('Real LongLive R1 backward requires CUDA')
    device = torch.device('cuda')
    torch.manual_seed(config['seed'])
    dataset = make_dataset(config, 'train')
    router_config = config['reality_memory']['router']
    bank = R1CandidateBank(dataset, router_config['reference_count'])
    index = 0  # Fixed train target; never select a case based on retrieval scores.
    batch = collate_reality([dataset[index]])
    candidates, refs, donor = bank[index]
    candidates = candidates.to(device)
    settings = config['reality_memory']
    encoder = RealityEncoder(ROOT / settings['encoder']['path'], settings['projector']['memory_dim'],
                             settings['projector']['num_memory_tokens'], settings['encoder']['image_size'])
    encoder = encoder.to(device).eval().requires_grad_(False)
    if encoder.identity != dataset.cache.identity:
        raise ValueError('Prefix and reference encoder identities differ')
    prefix, frame_ids = encode_prefix(encoder, batch['pixels'].to(device),
                                     settings['objective']['prefix_latents'], router_config['prefix_frames'])
    global_async = None
    if args.branch == 'global_async':
        from restream.reality_paired import load_global_constant
        global_async = load_global_constant(config, dataset.cache, device, role='async')
    features, mask, routing = select_r1_memory(args.branch, prefix, candidates, router_config['reference_count'],
                                              global_async, router_config['temperature'])
    assert not features.requires_grad and all(p.grad is None for p in encoder.parameters())
    if args.branch == 'routed':
        selected = int(routing['selected_indices'][0, 0])
        torch.testing.assert_close(features[0, 0], candidates[0, selected], rtol=0, atol=0)
    del encoder
    print('Loading frozen LongLive for one R1-A forward/backward', flush=True)
    pipeline = load_pipeline(config, device)
    # Independent of encoder/backbone initialization RNG, identical for all branches.
    torch.manual_seed(config['seed'])
    memory = make_memory(config).to(device)
    initial_digest = hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in memory.parameters())).hexdigest()
    gt, conditioning, anchor = prepare_reality(pipeline, batch, device, config)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        initial, _ = memory(conditioning['prompt_embeds'], features, mask)
        absent, _ = memory(conditioning['prompt_embeds'], features, torch.zeros_like(mask))
        assert torch.equal(initial, conditioning['prompt_embeds'])
        assert torch.equal(absent, conditioning['prompt_embeds'])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        fused, stats = memory(conditioning['prompt_embeds'], features, mask)
        loss = future_loss(pipeline, PreserveHistory(), gt, gt[:, :anchor + 1], gt[:, anchor:anchor + 1],
                           {**conditioning, 'prompt_embeds': fused}, anchor,
                           torch.Generator(device=device).manual_seed(123), 0)
    assert torch.isfinite(loss)
    print(f'Real R1 future loss={loss.item():.6f}; running backward', flush=True)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters())
    norms = {name: p.grad.float().norm().item() for name, p in memory.named_parameters()}
    assert norms['output.weight'] > 0
    assert all(not p.requires_grad and p.grad is None for p in pipeline.parameters())
    assert initial_digest == hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in memory.parameters())).hexdigest()
    report = {
        'status': 'passed', 'branch': args.branch, 'config': config, 'optimizer_steps': 0,
        'forward_backward_steps': 1, 'adapter_initialization': 'fresh_zero_init',
        'initial_adapter_sha256': initial_digest, 'noise_seed': 123, 'regularization_weights': [0, 0],
        'sample_id': batch['sample_id'][0], 'hard_donor_sample_id': dataset.rows[donor]['sample_id'],
        'prefix_frame_indices': frame_ids, 'prefix_frame_times': batch['sampled_times'][0, frame_ids].tolist(),
        'visible_until': dataset.rows[index]['visible_until'],
        'candidate_references': refs, 'routing': {k: v.detach().cpu().tolist() for k, v in routing.items()},
        'loss': loss.item(), 'parameter_grad_norms': norms,
        'initial_context_exact_base': True, 'no_memory_context_exact_base': True,
        'backbone_gradients_none': True, 'router_and_encoder_frozen': True,
        'train_manifest_sha256': manifest_digest(ROOT / config['data']['train_manifest']),
        'dev_manifest_sha256': manifest_digest(ROOT / config['data']['val_manifest']),
        'split_lock_sha256': manifest_digest(ROOT / config['data']['r1_split_lock']),
        'selection_config_hash': selection_config_hash(config), 'encoder_identity': dataset.cache.identity,
        'gpu': torch.cuda.get_device_name(), 'peak_vram_bytes': torch.cuda.max_memory_allocated(),
        'torch_version': torch.__version__,
        'scope': 'Gradient connectivity only. Zero-init blocks upstream adapter gradients on the first backward; output.weight must receive nonzero video-loss gradients.',
    }
    write_json(args.output, report)
    print(f"PASS {args.branch}: loss={loss.item():.6f}, output.weight grad={norms['output.weight']:.6f}, optimizer_steps=0", flush=True)


if __name__ == '__main__':
    main()
