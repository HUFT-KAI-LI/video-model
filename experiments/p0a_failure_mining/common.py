"""CPU-only contracts shared by P0-A commands. Block/frame indices are zero-based."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAXONOMY = ('identity_drift', 'clothing_color_drift', 'clothing_texture_drift',
            'accessory_disappearance', 'accessory_color_drift', 'object_identity_drift',
            'object_color_drift', 'structural_collapse', 'other')
REVIEW_FIELDS = ['video_id', 'selection', 'failure_confirmed', 'failure_type',
                 'failure_onset_block', 'failure_onset_frame', 'onset_kind',
                 'pre_onset_normal', 'reviewed_until_sec', 'human_confidence', 'notes']


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def geometry(duration, fps=16, block_size=3):
    if duration <= 0 or fps <= 0 or block_size != 3:
        raise ValueError('Positive duration/fps and official 3-latent blocks required')
    pixel_frames = math.ceil(duration * fps)
    latent_frames = math.ceil((pixel_frames + 3) / 4 / block_size) * block_size
    blocks = []
    for b in range(latent_frames // block_size):
        start = max(0, 4 * b * block_size - 3)
        end = min(pixel_frames, 4 * (b + 1) * block_size - 3)
        blocks.append(dict(block_index=b, latent_start=b * block_size,
                           latent_end=(b + 1) * block_size, frame_start=start,
                           frame_end=end, start_sec=start / fps, end_sec=end / fps))
    return dict(pixel_frames=pixel_frames, latent_frames=latent_frames,
                num_blocks=len(blocks), fps=fps, duration_sec=pixel_frames / fps,
                blocks=blocks)


def load_config(path):
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    if config['duration_sec'] not in (45, 60):
        raise ValueError('Pilot duration must be explicitly 45 or 60 seconds')
    if config['fps'] != 16 or (config['height'], config['width']) != (480, 832):
        raise ValueError('P0-A baseline is fixed at 480x832, 16 FPS')
    if len(set(config['seeds'])) != len(config['seeds']) or not config['seeds']:
        raise ValueError('Seeds must be nonempty and unique')
    if any(not isinstance(s, int) or s < 0 for s in config['seeds']):
        raise ValueError('Seeds must be nonnegative integers')
    mining = config['mining']
    if not 0 < mining['top_fraction'] <= 1 or not 0 <= mining['audit_fraction'] <= 1:
        raise ValueError('Invalid mining/audit fraction')
    if min(mining[k] for k in ('reference_blocks', 'persistence_blocks', 'change_window')) < 1:
        raise ValueError('Mining windows must be positive')
    return config


def load_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [r['prompt_id'] for r in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError('Prompt IDs must be unique and nonempty')
    for r in rows:
        if r['difficulty'] not in ('easy', 'medium', 'stress'):
            raise ValueError('Unknown difficulty')
        if not r['alternatives'] or r['attribute_text'] in r['alternatives']:
            raise ValueError('Distinct attribute alternatives required')
        if not r['prompt_id'].isalnum():
            raise ValueError('Prompt IDs must be alphanumeric')
    return rows


def build_plan(config, prompts):
    shape = geometry(config['duration_sec'], config['fps'])
    return [dict(**p, seed=s, video_id=f"{p['prompt_id']}_S{s:02d}",
                 duration_sec=shape['duration_sec'], num_blocks=shape['num_blocks'],
                 fps=config['fps'], num_frames=shape['pixel_frames'],
                 latent_frames=shape['latent_frames']) for p in prompts for s in config['seeds']]


def completed(run):
    run = Path(run)
    plan = json.loads((run / 'plan.json').read_text())
    result = []
    for row in plan['videos']:
        path = run / 'outputs' / row['video_id'] / 'complete.json'
        if path.exists():
            data = json.loads(path.read_text())
            if data['plan_signature'] != plan['signature']:
                raise ValueError(f'Stale output: {path}')
            result.append(data)
    return result
