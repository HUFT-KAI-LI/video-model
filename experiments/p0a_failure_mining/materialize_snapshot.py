#!/usr/bin/env python3
"""Reproduce a baseline prefix, require bitwise equality, then capture exact caches. No branching."""
import argparse
import json
from pathlib import Path
from common import digest, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--video-id', required=True)
    p.add_argument('--after-block', type=int, required=True, help='Zero-based pre-onset block')
    args = p.parse_args()
    import torch
    from backend import cpu_tree, load, provenance, rng_state, set_rng
    run = args.run_dir.resolve()
    plan = json.loads((run / 'plan.json').read_text())
    if args.video_id not in {v['video_id'] for v in plan['videos']}:
        raise ValueError('Unknown video ID')
    out = run / 'outputs' / args.video_id
    manifest = json.loads((out / 'complete.json').read_text())
    if manifest['plan_signature'] != plan['signature'] or digest(out / 'replay.pt') != manifest['artifact_sha256']['replay.pt']:
        raise ValueError('Replay capsule provenance mismatch')
    if provenance(plan['config']) != json.loads((run / 'provenance.json').read_text()):
        raise ValueError('Exact replay requires identical models, source and runtime')
    capsule = torch.load(out / 'replay.pt', map_location='cpu', weights_only=True)
    if not 0 <= args.after_block < capsule['row']['num_blocks'] - 1:
        raise ValueError('Need a valid pre-onset block with a future')
    target = out / f'snapshot_after_{args.after_block:03d}.pt'
    if target.exists():
        raise FileExistsError(target)
    pipe = load(plan['config'])
    class Captured(Exception):
        pass
    def observer(block, prefix, pipeline):
        if not torch.equal(prefix.cpu(), capsule['latents'][:, :prefix.shape[1]]):
            raise RuntimeError(f'Replay diverged at block {block}; state is NOT eligible for P0-B')
        state = rng_state()
        expected = capsule['boundary_rng'][block]
        if not torch.equal(state['cpu'], expected['cpu']) or any(not torch.equal(a, b) for a, b in zip(state['cuda'], expected['cuda'])):
            raise RuntimeError('Replay RNG diverged')
        if block == args.after_block:
            torch.save(dict(kv_cache=cpu_tree(pipeline.kv_cache1),
                            crossattn_cache=cpu_tree(pipeline.crossattn_cache), rng=state,
                            next_latent_frame=prefix.shape[1], prefix_latents=prefix.cpu().clone(),
                            future_noise=capsule['noise'][:, prefix.shape[1]:].clone(),
                            prompt=capsule['row']['prompt'], plan_signature=plan['signature']), target)
            write_json(target.with_suffix('.json'), dict(video_id=args.video_id, after_block=block,
                       verified=True, snapshot_sha256=digest(target),
                       replay_sha256=manifest['artifact_sha256']['replay.pt'], plan_signature=plan['signature']))
            raise Captured()
    with torch.inference_mode():
        noise = capsule['noise'].to('cuda')
        set_rng(capsule['initial_rng'])
        try:
            pipe.inference(noise, [capsule['row']['prompt']], low_memory=True,
                           block_callback=observer, decode_video=False)
        except Captured:
            print(target)


if __name__ == '__main__':
    main()
