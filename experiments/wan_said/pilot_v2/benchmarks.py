import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from ..data import caption_key, load_split
from ..io_utils import read_jsonl, sha256, write_json, write_jsonl
from ..runtime import ROOT
from .config import SOURCE_LOCK, digest, load_config

T2V = [('consistent_attr', '1_consistent_attr.txt'), ('dynamic_attr', '2_dynamic_attr.txt'),
       ('spatial_relationships', '3_spatial_relationship.txt'), ('motion_binding', '4_motion_binding.txt'),
       ('action_binding', '5_action_binding.txt'), ('interaction', '6_interaction.txt'),
       ('generative_numeracy', '7_numeracy.txt')]
T2V_METADATA = dict(consistent_attr='consistent_attribute_binding', dynamic_attr='dynamic_attribute_binding',
                    spatial_relationships='spatial_relationships', motion_binding='motion_binding',
                    action_binding='action_binding', interaction='object_interactions',
                    generative_numeracy='generative_numeracy')
SEEDS = [20261021, 20261022, 20261023, 20261024, 20261025]


def check_vendor(name, lock_key=None):
    path = ROOT/'vendor'/name
    expected = SOURCE_LOCK[lock_key or name]['commit']
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError(f'Benchmark source revision mismatch: {name}')
    return path


def source_hash(repo, path):
    relative = str(path.relative_to(repo))
    expected = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD:'+relative], text=True).strip()
    actual = subprocess.check_output(['git', '-C', str(repo), 'hash-object', relative], text=True).strip()
    if actual != expected:
        raise ValueError(f'Pinned benchmark source file was modified: {path}')
    return sha256(path)


def suites():
    comp = check_vendor('T2V-CompBench')
    vb = check_vendor('VBench')
    repa = check_vendor('VideoREPA')
    sources, sets = {}, {}
    items, metadata_differences = [], []
    for category, name in T2V:
        path = comp/'prompts'/name
        prompts = path.read_text().splitlines()
        if len(prompts) != 200 or not all(p.strip() for p in prompts):
            raise ValueError(f'T2V-CompBench category changed: {name}')
        sources[str(path.relative_to(ROOT))] = source_hash(comp, path)
        metadata_path = comp/'meta_data'/(T2V_METADATA[category]+'.json')
        sources[str(metadata_path.relative_to(ROOT))] = source_hash(comp, metadata_path)
        category_metadata = json.loads(metadata_path.read_text())
        if len(category_metadata) != len(prompts):
            raise ValueError(f'Official T2V prompt/metadata row count differs: {category}')
        metadata_differences.extend(dict(category=category, row=i, generation_prompt=p, metadata_prompt=r['prompt'])
                                    for i, (r, p) in enumerate(zip(category_metadata, prompts), 1)
                                    if caption_key(r['prompt']) != caption_key(p))
        for i, text in enumerate(prompts, 1):
            items.append(dict(id=f'{category}_{i:04d}', category=category,
                              prompt=text.strip(), number=i))
    sets['t2v_compbench'] = items
    path = vb/'vbench/VBench_full_info.json'
    sources[str(path.relative_to(ROOT))] = source_hash(vb, path)
    metadata = json.loads(path.read_text())
    merged = {}
    for i, row in enumerate(metadata):
        prompt = row['prompt_en']
        if '/' in prompt or len((prompt+'-0.mp4').encode()) > 255:
            raise ValueError('Official VBench filename cannot be represented safely on this filesystem')
        if prompt not in merged:
            merged[prompt] = dict(id=digest(prompt)[:20], prompt=prompt, dimensions=[], source_rows=[])
        item = merged[prompt]
        item['source_rows'].append(i)
        item['dimensions'] = sorted(set(item['dimensions']) | set(row['dimension']))
    sets['vbench'] = list(merged.values())
    path = repa/'evaluation/VIDEOPHY2/videophy2.csv'
    alternate = repa/'evaluation/VIDEOPHY2/prompt-upsampled-test.csv'
    sources[str(path.relative_to(ROOT))] = source_hash(repa, path)
    sources[str(alternate.relative_to(ROOT))] = source_hash(repa, alternate)
    with path.open(newline='') as stream:
        phy = list(csv.DictReader(stream))
    with alternate.open(newline='') as stream:
        phy_alternate = list(csv.DictReader(stream))
    if not all(r['caption'].strip() and r['upsampled_caption'].strip() for r in phy):
        raise ValueError('Empty VideoPhy2 original/upsampled prompt')
    sets['videophy2'] = [dict(id=f'videophy2_{i:04d}', number=i, prompt=r['caption'],
                              upsampled_prompt=r['upsampled_caption']) for i, r in enumerate(phy, 1)]
    audit = dict(source_files_sha256=sources,
                 source_commits={key: SOURCE_LOCK[key]['commit'] for key in ('T2V-CompBench', 'VBench', 'VideoREPA')},
                 t2v_compbench=dict(count=len(items), categories=dict(Counter(i['category'] for i in items)),
                                    normalized_prompt_metadata_differences=metadata_differences,
                                    mapping_policy='Preserve official original prompts and official metadata unchanged; record source wording discrepancies'),
                 vbench=dict(official_metadata_rows=len(metadata), unique_generation_prompts=len(merged),
                             dimensions=sorted({d for r in metadata for d in r['dimension']})),
                 videophy2=dict(pinned_csv_rows=len(phy), alternate_csv_rows=len(phy_alternate),
                                benchmark_version='VideoPhy-2', expected_test_prompts=590,
                                rating_scale=[1, 5], primary_metric='joint_at_4',
                                joint_condition='SA >= 4 AND PC >= 4 on the same video',
                                separate_early_benchmark=dict(name='VideoPhy', test_prompts=344,
                                    evaluator='VideoCon-Physics', ratings='binary SA/PC', comparable=False),
                                policy='590 VideoPhy-2 prompts; never truncate or compare directly with early VideoPhy',
                                default_prompt_variant='original', main_table_ready=False))
    return sets, audit


def make_jobs(suite, prompts, upsampled=False):
    jobs = []
    for row in prompts:
        seeds = SEEDS if suite == 'vbench' else SEEDS[:1]
        for seed_index, seed in enumerate(seeds):
            if suite == 'vbench':
                relative = f'vbench/{row["prompt"]}-{seed_index}.mp4'
            elif suite == 't2v_compbench':
                relative = f't2v_compbench/{row["category"]}/{row["number"]:04d}.mp4'
            elif suite == 'videophy2':
                relative = f'videophy2/{row["number"]:04d}.mp4'
            else:
                raise ValueError(suite)
            jobs.append(dict(job_id=digest([suite, row['id'], seed, upsampled])[:24],
                             suite=suite, prompt_id=row['id'], seed=seed,
                             prompt=row['upsampled_prompt'] if upsampled else row['prompt'],
                             original_prompt=row['prompt'], relative_path=relative,
                             prompt_variant='upsampled' if upsampled else 'original'))
    return jobs


def prepare(cfg, output):
    output = Path(output)
    sets, audit = suites()
    all_captions = {caption_key(r['caption']) for split in ('train', 'val', 'test')
                    for r in load_split(cfg['manifests'], split)}
    overlap = [dict(suite=name, id=r['id']) for name, rows in sets.items() for r in rows
               if caption_key(r['prompt']) in all_captions or
               (r.get('upsampled_prompt') and caption_key(r['upsampled_prompt']) in all_captions)]
    if overlap:
        raise ValueError(f'Benchmark caption overlaps a pilot split: {overlap}')
    output.mkdir(parents=True, exist_ok=True)
    plans = {}
    for name, rows in sets.items():
        jobs = make_jobs(name, rows)
        path = output/f'{name}.jsonl'
        if path.exists() and read_jsonl(path) != jobs:
            raise ValueError('Frozen benchmark plan changed; use another output directory')
        write_jsonl(path, jobs)
        plans[name] = dict(jobs=len(jobs), plan_sha256=sha256(path))
    audit.update(plans=plans, split_overlap=overlap, official_scorers_executed=False,
                 no_proxy_metrics=True, formal_generation=dict(frames=81, height=480, width=832, fps=16,
                 sampler='official Wan FlowUniPCMultistepScheduler', steps=50, cfg=6, shift=8,
                 prompt_extension=False, seeds=SEEDS))
    write_json(output/'suite_lock.json', audit)
    return audit


def validate_job_paths(jobs):
    ids, paths = set(), set()
    for job in jobs:
        if (not re.fullmatch('[a-f0-9]{24}', job['job_id']) or
                not isinstance(job['seed'], int) or not job['prompt'].strip()):
            raise ValueError('Invalid benchmark job identity, seed, or prompt')
        path = Path(job['relative_path'])
        if path.is_absolute() or '..' in path.parts or path.suffix != '.mp4':
            raise ValueError('Unsafe benchmark output path')
        if job['job_id'] in ids or job['relative_path'] in paths:
            raise ValueError('Duplicate benchmark job/video mapping')
        ids.add(job['job_id'])
        paths.add(job['relative_path'])


def validate_official_plan(jobs):
    if not jobs:
        raise ValueError('An empty benchmark is not a complete official plan')
    suite = jobs[0]['suite']
    sets, _ = suites()
    if suite not in sets or jobs != make_jobs(suite, sets[suite]):
        raise ValueError('Formal scoring/generation requires the complete pinned official prompt/seed plan')
    validate_job_paths(jobs)


def coverage(jobs, video_root):
    validate_job_paths(jobs)
    missing = [j['job_id'] for j in jobs if not (Path(video_root)/j['relative_path']).is_file()]
    return dict(expected=len(jobs), available=len(jobs)-len(missing), missing_job_ids=missing,
                denominator=len(jobs), complete=not missing)


def require_complete(jobs, video_root):
    result = coverage(jobs, video_root)
    if not result['complete']:
        raise ValueError(f'Missing {len(result["missing_job_ids"])} / {result["denominator"]} videos; refusing a reduced denominator')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(load_config(args.config), args.output), indent=2))
