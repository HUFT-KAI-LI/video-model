"""One sequential worker per GPU, with atomic shared job claims and no quality retries."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from protocol import ROOT, HERE, DESIGN, MODELS, jobs, freeze, digest, write_json


def load_model(name):
    import torch
    sys.path.insert(0, str(ROOT / 'restream_mvp/code/LongLive'))
    if name == 'wan':
        sys.path.insert(0, str(HERE / 'vendor/wan21'))
        from wan.text2video import WanT2V
        from wan.configs import WAN_CONFIGS
        pipe = WanT2V(WAN_CONFIGS['t2v-1.3B'], str(ROOT / 'restream_mvp/models/Wan2.1-T2V-1.3B'))
        # Keep text encoder resident during denoising. Free it only for the FP32 VAE peak.
        original_decode = pipe.vae.decode
        def decode(latents):
            pipe.text_encoder.model.cpu()
            torch.cuda.empty_cache()
            return original_decode(latents)
        pipe.vae.decode = decode
        return pipe
    sys.path.insert(0, str(ROOT / 'restream_mvp'))
    from restream.runtime import load_pipeline
    return load_pipeline({'model': dict(wan='models/Wan2.1-T2V-1.3B',
                         longlive='models/LongLive-1.3B',
                         upstream_config='code/LongLive/configs/longlive_inference.yaml',
                         component_devices=dict(text_encoder='cuda', vae='cuda')),
                         'data': DESIGN}, torch.device('cuda'))


def write_video(path, chunks):
    import av
    import torch
    count = 0
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('libx264', rate=DESIGN['fps'])
        stream.width, stream.height = DESIGN['width'], DESIGN['height']
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18'}
        for pixels in chunks:
            arrays = ((pixels * .5 + .5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for array in arrays:
                if count >= DESIGN['saved_frames']:
                    break
                for packet in stream.encode(av.VideoFrame.from_ndarray(array, format='rgb24')):
                    out.mux(packet)
                count += 1
        for packet in stream.encode():
            out.mux(packet)
    if count != DESIGN['saved_frames']:
        raise RuntimeError(f'Frame count mismatch: {count}')
    return count


def generate(pipe, job, folder):
    import numpy as np
    import torch
    seed = job['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad():
        if job['model'] == 'wan':
            pixels = pipe.generate(job['prompt'], size=(DESIGN['width'], DESIGN['height']),
                                   frame_num=DESIGN['native_frames'], shift=DESIGN['wan']['shift'],
                                   sample_solver='unipc', sampling_steps=DESIGN['wan']['steps'],
                                   guide_scale=DESIGN['wan']['cfg'], seed=seed, offload_model=False)
            return write_video(folder / 'video.mp4', [pixels.permute(1, 0, 2, 3)])
        # Release previous caches before a new allocation, otherwise peak memory doubles.
        pipe.kv_cache1 = None
        pipe.crossattn_cache = None
        noise = torch.randn(1, DESIGN['latent_frames'], 16, DESIGN['height']//8,
                            DESIGN['width']//8, device='cuda', dtype=torch.bfloat16)
        latent = pipe.inference(noise, [job['prompt']], low_memory=False, decode_video=False)
        torch.save(dict(noise=noise.cpu(), latents=latent.cpu()), folder / 'latents.pt')
        pipe.kv_cache1 = None
        pipe.crossattn_cache = None
        torch.cuda.empty_cache()
        pipe.vae.model.clear_cache()
        def chunks():
            for start in range(0, DESIGN['latent_frames'], 3):
                yield pipe.vae.decode_to_pixel(latent[:, start:start+3], use_cache=True)[0]
        try:
            return write_video(folder / 'video.mp4', chunks())
        finally:
            pipe.vae.model.clear_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--model', choices=MODELS, required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA required')
    run = args.run.resolve()
    plan_id = freeze(run)
    import fcntl
    import numpy as np
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    lock = (run / f'.gpu_{gpu}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = run / 'sources'
    source.mkdir(exist_ok=True)
    # Archive exact generation sources before model loading; evaluation may evolve separately.
    import shutil
    for rel in json.loads((run / 'plan.json').read_text())['generation_source_sha256']:
        dest = source / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(ROOT / rel, dest)
    worker = f'{args.model}_gpu{gpu}_{os.getpid()}'
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    pipe = load_model(args.model)
    torch.cuda.synchronize()
    write_json(run / f'{worker}_load.json', dict(seconds=time.time()-started,
               device=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
               plan_id=plan_id, gpu=gpu, model=args.model))
    count = 0
    for job in jobs(args.model):
        folder = run / 'outputs' / job['video_id']
        if (folder / 'complete.json').exists():
            saved = json.loads((folder / 'complete.json').read_text())
            if saved['plan_id'] != plan_id or digest(folder / 'video.mp4') != saved['video_sha256']:
                raise RuntimeError(f'Invalid existing result: {folder}')
            continue
        folder.parent.mkdir(parents=True, exist_ok=True)
        try:
            folder.mkdir()
        except FileExistsError:
            # Claimed by another GPU, or an explicitly preserved failed attempt.
            continue
        write_json(folder / 'started.json', dict(job=job, worker=worker, plan_id=plan_id, started=time.time()))
        print(f"START {job['video_id']}", flush=True)
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        try:
            frames = generate(pipe, job, folder)
            torch.cuda.synchronize()
            manifest = dict(job=job, plan_id=plan_id, worker=worker, frames=frames,
                            seconds=time.time()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                            video_sha256=digest(folder / 'video.mp4'))
            if (folder / 'latents.pt').exists():
                manifest['latents_sha256'] = digest(folder / 'latents.pt')
            write_json(folder / 'complete.json', manifest)
            print('COMPLETE ' + json.dumps(manifest), flush=True)
        except Exception:
            write_json(folder / 'failed.json', dict(traceback=traceback.format_exc(), worker=worker,
                                                   seconds=time.time()-start, plan_id=plan_id))
            raise
        count += 1
        gc.collect()
        if args.limit and count >= args.limit:
            break
    print(f'WORKER FINISHED {worker}: {count} new videos', flush=True)


if __name__ == '__main__':
    main()
