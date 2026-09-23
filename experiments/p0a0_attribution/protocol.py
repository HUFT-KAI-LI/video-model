"""Frozen P0-A0 design and artifact helpers; no GPU dependency."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments/p0a_failure_mining'))
from common import digest, signature, write_json, write_csv, read_csv

COLORS = ['red', 'blue', 'yellow', 'green', 'orange', 'black', 'white', 'brown', 'pink', 'purple', 'gray']
MODELS = ['wan', 'longlive']
DESIGN = dict(experiment='P0-A0', seeds=[0, 1, 2, 3], height=480, width=832,
              fps=16, duration_sec=5, native_frames=81, saved_frames=80, latent_frames=21,
              initial_frames=list(range(16)), proxy_frames=[0, 4, 8, 12],
              secondary_proxy_frames=[16, 32, 48, 64, 79],
              colors=COLORS, primary='human_initial_attribute_accuracy',
              primary_rule='CORRECT iff target entity/item is present with specified color throughout the visible portions of frames 0..15; INCORRECT if visibly missing/wrong; UNJUDGEABLE if occluded/ambiguous; absent labels remain pending',
              wan=dict(steps=50, solver='unipc', shift=8.0, cfg=6.0,
                       negative_prompt='official wan.configs.shared_config default',
                       transformer_parameter_dtype='float32', transformer_autocast='bfloat16', vae_dtype='float32'),
              longlive=dict(steps=[1000, 750, 500, 250], shift=5.0, cfg=None,
                            negative_prompt=None, local_attn_size=12, sink_size=3,
                            latent_block_size=3, official_lora=True, dtype='bfloat16'),
              prompt_extension=False, seed_pairing='same integer, NOT identical noise tensors or RNG streams',
              inference_scope='two released inference pipelines; NOT isolated causalization effect',
              blind_order_seed=20260923)


def prompts():
    path = ROOT / 'experiments/p0a_failure_mining/prompts.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(rows) == 20 and len({r['prompt_id'] for r in rows}) == 20
    return rows


def jobs(model):
    return [dict(p, model=model, seed=s, pair_id=f"{p['prompt_id']}_S{s:02d}",
                 video_id=f"{model}_{p['prompt_id']}_S{s:02d}") for p in prompts() for s in DESIGN['seeds']]


def color_texts(prompt):
    # Replace the ATTRIBUTE color only: e.g. retain 'white dog' in every harness contrast.
    target = next(c for c in COLORS if c in prompt['target_attribute'].split())
    text = prompt['attribute_text']
    index = text.rfind(' ' + target + ' ')
    if index < 0:
        raise ValueError(text)
    return target, [text[:index + 1] + c + text[index + len(target) + 1:] for c in COLORS]


def source_hashes():
    paths = [HERE / 'generate.py', HERE / 'protocol.py', ROOT / 'restream_mvp/restream/runtime.py']
    paths += [p for p in (HERE / 'vendor/wan21').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [p for p in (ROOT / 'restream_mvp/code/LongLive').rglob('*')
              if p.is_file() and p.suffix in ('.py', '.yaml')]
    paths += [p for p in (HERE / 'vendor/wan21').rglob('*') if p.is_file() and p.suffix in ('.py', '.json')]
    paths += [ROOT / 'experiments/p0a_failure_mining/common.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def make_plan():
    return dict(design=DESIGN, prompts=prompts(), jobs=[j for m in MODELS for j in jobs(m)],
                generation_source_sha256=source_hashes())


def freeze(run):
    import fcntl
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.plan.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = make_plan()
        path = run / 'plan.json'
        if path.exists():
            if json.loads(path.read_text()) != plan:
                raise RuntimeError('Frozen design, prompts or generation sources changed; use a new run directory')
        else:
            write_json(path, plan)
        return signature(plan)
