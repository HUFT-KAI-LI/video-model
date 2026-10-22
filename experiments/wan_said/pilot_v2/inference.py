"""Native Wan inference plus a pinned official Wan UniPC solver; no auxiliary teacher."""
import argparse
import ast
import fcntl
import importlib.util
import json
import subprocess
import shutil
import time
from pathlib import Path

import av
import imageio.v2 as imageio
import torch

from ..data import caption_key, load_split
from ..io_utils import read_jsonl, sha256, write_json
from ..runtime import forward_fn, load_dit, load_text, load_trainable, load_vae
from .benchmarks import validate_job_paths, validate_official_plan
from .config import load_config, signature

WAN_COMMIT = '9737cba9c1c3c4d04b33fcad41c111989865d315'
WAN_SOURCE_SHA256 = {
    'wan/utils/fm_solvers_unipc.py': '0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9',
    'wan/configs/shared_config.py': 'a942b6f0a1d1a4ec9a7893c17a6218651ce272953b90d012ce5cdd19d15e45f7',
}


def native_generation_protocol():
    repo = Path(__file__).resolve().parents[1]/'vendor/Wan2.1'
    if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != WAN_COMMIT:
        raise ValueError('Wan inference source revision changed')
    for name, expected in WAN_SOURCE_SHA256.items():
        if sha256(repo/name) != expected:
            raise ValueError(f'Pinned Wan inference source was modified: {name}')
    shared = repo/'wan/configs/shared_config.py'
    negative = None
    for node in ast.parse(shared.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'sample_neg_prompt' for t in node.targets):
            negative = ast.literal_eval(node.value)
    if not negative:
        raise ValueError('Pinned official negative prompt was not found')
    return dict(frames=81, height=480, width=832, fps=16, steps=50, cfg=6., shift=8.,
                sampler='official_wan_unipc', prompt_extension=False, negative_prompt=negative,
                wan_source_commit=WAN_COMMIT,
                solver_sha256=sha256(repo/'wan/utils/fm_solvers_unipc.py'),
                negative_source_sha256=sha256(shared))


def solver(protocol, device):
    path = Path(__file__).resolve().parents[1]/'vendor/Wan2.1/wan/utils/fm_solvers_unipc.py'
    if sha256(path) != protocol['solver_sha256']:
        raise ValueError('Official Wan UniPC solver file changed')
    spec = importlib.util.spec_from_file_location('wan_said_pilot_native_unipc', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = module.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(protocol['steps'], device=device, shift=protocol['shift'])
    return scheduler


def verify_video(path, protocol):
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        if (stream.height, stream.width) != (protocol['height'], protocol['width']):
            raise ValueError('Generated video dimensions differ from the protocol')
        if abs(float(stream.average_rate)-protocol['fps']) > .01:
            raise ValueError('Generated FPS differs from the protocol')
        if sum(1 for _ in video.decode(video=0)) != protocol['frames']:
            raise ValueError('Generated frame count differs from the protocol')


def worker_jobs(jobs, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError('Invalid generation worker index/count')
    return jobs[index::count]


def lock_generation(output, contract):
    # Four independent generators share a contract, but never a temporary JSON file.
    with (output/'generation.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        path = output/'generation_lock.json'
        if path.exists():
            if json.loads(path.read_text()) != contract:
                raise ValueError('Generation directory belongs to another model/seed/protocol/plan')
        else:
            write_json(path, contract)


@torch.no_grad()
def generate(cfg, jobs_path, output, group='base', checkpoint=None, limit=None, smoke=False,
             smoke_frames=17, smoke_steps=4, exploratory=False, worker_index=0, worker_count=1,
             minimum_free_gib=50):
    if group not in ('base', 'A', 'B', 'C') or (group == 'base') != (checkpoint is None):
        raise ValueError('Base uses no adapter; A/B/C require their corresponding checkpoint')
    if limit is not None and not smoke:
        raise ValueError('A partial plan may only be generated as an explicitly marked smoke run')
    jobs = read_jsonl(jobs_path)
    if not jobs or (limit is not None and limit < 1):
        raise ValueError('Generation requires a nonempty plan and a positive smoke limit')
    validate_job_paths(jobs)
    if not smoke:
        validate_official_plan(jobs)
    selected = worker_jobs(jobs[:limit] if limit is not None else jobs, worker_index, worker_count)
    protocol = native_generation_protocol()
    if smoke:
        if smoke_frames < 5 or (smoke_frames-1) % 4 or smoke_steps < 2:
            raise ValueError('Invalid inference smoke frame or step count')
        protocol.update(frames=smoke_frames, height=256, width=384, steps=smoke_steps)
    base_weights = {name: sha256(Path(cfg['wan_weights'])/name) for name in
                    ('diffusion_pytorch_model.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth')}
    checkpoint_state = torch.load(checkpoint, map_location='cpu', weights_only=True) if checkpoint else None
    if checkpoint_state is not None:
        identity = checkpoint_state['contract']['identity']
        if (checkpoint_state['group'] != group or identity['preprocessing_hash'] != signature(cfg) or
                identity['split_lock_sha256'] != sha256(Path(cfg['manifests'])/'split_lock.json') or
                identity['wan_weights'] != base_weights):
            raise ValueError('Inference adapter belongs to a different arm/teacher protocol/split')
        if checkpoint_state['contract']['smoke'] and not smoke:
            raise ValueError('A smoke adapter cannot be presented as a formal benchmark model')
        if bool(checkpoint_state['contract'].get('exploratory')) != exploratory:
            raise ValueError('Exploratory checkpoints require explicit --exploratory inference; no relabeling')
    training_captions = {r['caption_group'] for r in load_split(cfg['manifests'], 'train')}
    if any(caption_key(job['prompt']) in training_captions for job in jobs):
        raise ValueError('Benchmark prompt overlaps training captions')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(group=group, checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
                    jobs_sha256=sha256(jobs_path), protocol=protocol, smoke=smoke, base_weights=base_weights)
    if exploratory:
        contract.update(exploratory=True, publication_eligible=False,
                        training_review_status=checkpoint_state['contract']['review_status'] if checkpoint_state else None)
    lock_generation(output, contract)
    device = cfg['devices']['student']
    text_device = cfg['devices']['text']
    text, tokenizer = load_text(cfg, text_device)
    dit = load_dit(cfg, device, lora=group != 'base').eval()
    if checkpoint_state:
        load_trainable(dit, checkpoint_state['lora'])
    vae = load_vae(cfg, cfg['devices']['vae'])
    native = forward_fn()

    def context(prompt):
        ids, valid = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = text(ids.to(text_device), valid.to(text_device))
        return (hidden*valid.to(text_device)[..., None]).to(device)

    negative = context(protocol['negative_prompt'])
    completed, failures = [], []

    def progress():
        name = 'progress.json' if worker_count == 1 else f'progress.worker_{worker_index}.json'
        write_json(output/name, dict(expected=len(jobs), requested_this_run=len(selected),
                   completed_this_run=len(completed), failures=failures, smoke=smoke,
                   complete=(len(completed) == len(jobs) and not failures), denominator=len(jobs),
                   worker_index=worker_index, worker_count=worker_count,
                   worker_complete=(len(completed) == len(selected) and not failures)))

    for job in selected:
        path = output/job['relative_path']
        report = output/'records'/(job['job_id']+'.json')
        if path.exists():
            if not report.exists():
                raise ValueError('Unrecorded generated video; use a fresh output directory')
            saved = json.loads(report.read_text())
            if saved['job'] != job or saved['sha256'] != sha256(path):
                raise ValueError('Existing generation identity/checksum changed')
            verify_video(path, protocol)
            completed.append(saved)
            progress()
            continue
        start = time.monotonic()
        try:
            if shutil.disk_usage(output).free < minimum_free_gib*1024**3:
                raise OSError(f'Generation stopped to retain {minimum_free_gib} GiB free disk')
            generator = torch.Generator(device=device).manual_seed(job['seed'])
            shape = (1, 16, (protocol['frames']-1)//4+1, protocol['height']//8, protocol['width']//8)
            latent = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
            positive = context(job['prompt'])
            schedule = solver(protocol, device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for t in schedule.timesteps:
                    timestep = t[None].to(device=device, dtype=torch.bfloat16)
                    pos = native(dit=dit, latents=latent.to(torch.bfloat16), timestep=timestep, context=positive)
                    neg = native(dit=dit, latents=latent.to(torch.bfloat16), timestep=timestep, context=negative)
                    prediction = neg+protocol['cfg']*(pos-neg)
                    latent = schedule.step(prediction, t, latent, return_dict=False, generator=generator)[0]
                decoded = vae.decode(latent.to(device=cfg['devices']['vae'], dtype=torch.bfloat16),
                                     device=cfg['devices']['vae'], tiled=True)
            frames = ((decoded[0].float().cpu().permute(1, 2, 3, 0)+1)*127.5).clamp(0, 255).byte().numpy()
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(path, frames, fps=protocol['fps'], codec='libx264', quality=8)
            verify_video(path, protocol)
            saved = dict(job=job, sha256=sha256(path), path=str(path.resolve()), seconds=time.monotonic()-start,
                         pixel_std=float(frames.std()), teacher_used=False, auxiliary_modules_used=False,
                         temporal_pixel_delta=float(abs(frames[1:].astype('float32')-frames[:-1]).mean()),
                         formal_efficacy_evidence=False)
            write_json(report, saved)
            completed.append(saved)
            print(group, job['job_id'], saved['seconds'], flush=True)
        except Exception as error:
            failures.append(dict(job_id=job['job_id'], error=repr(error)))
            print(group, job['job_id'], repr(error), flush=True)
            if path.exists():
                path.rename(path.with_suffix(f'.failed.{time.time_ns()}.mp4'))
            progress()
            raise RuntimeError('Generation failed; resume only after fixing the recorded error') from error
        progress()
    if failures:
        raise RuntimeError(f'{len(failures)} generations failed; expected denominator is unchanged')
    return completed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--group', choices=['base', 'A', 'B', 'C'], default='base')
    parser.add_argument('--checkpoint')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-frames', type=int, default=17)
    parser.add_argument('--smoke-steps', type=int, default=4)
    parser.add_argument('--exploratory', action='store_true')
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--worker-count', type=int, default=1)
    parser.add_argument('--device', help='Place this independent generator on one CUDA device')
    parser.add_argument('--minimum-free-gib', type=float, default=50)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg['devices'] = dict.fromkeys(cfg['devices'], args.device)
    generate(cfg, args.jobs, args.output, args.group, args.checkpoint,
             args.limit, args.smoke, args.smoke_frames, args.smoke_steps, args.exploratory,
             args.worker_index, args.worker_count, args.minimum_free_gib)
