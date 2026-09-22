#!/usr/bin/env python3
"""Prepare a manifest by default. --execute runs natural trajectories, once each."""
import argparse
import json
from pathlib import Path
from common import HERE, build_plan, completed, digest, geometry, load_config, load_prompts, signature, write_csv, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=HERE / 'generation_config.yaml')
    p.add_argument('--prompts', type=Path, default=HERE / 'prompts.jsonl')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--video-id', action='append', help='Run only these IDs; may be repeated')
    args = p.parse_args()
    config, prompts = load_config(args.config), load_prompts(args.prompts)
    payload = {'config': config, 'videos': build_plan(config, prompts)}
    payload['signature'] = signature(payload)
    run = args.run_dir.resolve()
    plan_path = run / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != payload:
        raise ValueError('Run directory belongs to a different config/prompt pool; use a new directory')
    write_json(plan_path, payload)
    write_csv(run / 'manifests/planned_videos.csv', [{k: v for k, v in r.items() if k != 'alternatives'} for r in payload['videos']])
    write_json(run / 'block_map.json', geometry(config['duration_sec'], config['fps']))
    print(f"Prepared {len(payload['videos'])} trajectories at {run}")
    if not args.execute:
        return
    selected = args.video_id or [r['video_id'] for r in payload['videos']]
    if set(selected) - {r['video_id'] for r in payload['videos']}:
        raise ValueError('Unknown video ID')
    import torch
    from backend import load, provenance, rng_state, decode_video
    evidence = provenance(config)
    provenance_path = run / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != evidence:
        raise ValueError('Runtime/source/model provenance changed; use a new run directory')
    write_json(provenance_path, evidence)
    pipe = None
    for row in payload['videos']:
        if row['video_id'] not in selected:
            continue
        out = run / 'outputs' / row['video_id']
        if (out / 'complete.json').exists():
            saved = json.loads((out / 'complete.json').read_text())
            if saved['plan_signature'] != payload['signature']:
                raise ValueError('Completion manifest does not match plan')
            for name, sha in saved['artifact_sha256'].items():
                if not (out / name).is_file() or digest(out / name) != sha:
                    raise ValueError(f'Completed artifact missing or changed: {out / name}')
            print('Already complete:', row['video_id'])
            continue
        if out.exists():
            raise RuntimeError(f'Incomplete trajectory at {out}; no implicit retry. Archive it and use a new run directory.')
        if pipe is None:
            pipe = load(config)
        out.mkdir(parents=True)
        write_json(out / 'started.json', row)
        try:
            with torch.inference_mode():
                torch.manual_seed(row['seed'])
                torch.cuda.manual_seed_all(row['seed'])
                noise = torch.randn((1, row['latent_frames'], 16, config['height']//8, config['width']//8),
                                    device='cuda', dtype=torch.bfloat16)
                initial_rng = rng_state()
                boundary_rng = []
                def observer(block, prefix, pipeline):
                    boundary_rng.append(rng_state())
                latents = pipe.inference(noise, [row['prompt']], low_memory=True,
                                         block_callback=observer, decode_video=False)
                torch.save(dict(noise=noise.cpu(), latents=latents.cpu(), initial_rng=initial_rng,
                                boundary_rng=boundary_rng, row=row, plan_signature=payload['signature']),
                           out / 'replay.pt')
                del noise
                decode_video(pipe, latents, out / 'video.mp4', row['num_frames'], row['fps'])
            summary = dict(row, plan_signature=payload['signature'], status='complete',
                           snapshot_status='replay_available_not_verified',
                           artifact_sha256={name: digest(out / name) for name in ('video.mp4', 'replay.pt')})
            write_json(out / 'complete.json', summary)
            print('Completed:', row['video_id'], flush=True)
        except Exception as error:
            write_json(out / 'failed.json', {'error': repr(error), 'video_id': row['video_id']})
            raise
    rows = completed(run)
    if rows:
        write_csv(run / 'manifests/video_manifest.csv', [{k: v for k, v in r.items() if k not in ('artifact_sha256', 'alternatives')} for r in rows])


if __name__ == '__main__':
    main()
