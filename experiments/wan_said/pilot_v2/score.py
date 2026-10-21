"""Run pinned official scoring entry points without changing the expected denominator."""
import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from ..io_utils import read_jsonl, sha256, write_json
from ..runtime import ROOT
from .benchmarks import check_vendor, require_complete, source_hash, suites, validate_official_plan
from .inference import native_generation_protocol, verify_video


def recipes(suite, videos, output):
    videos, output = Path(videos).resolve(), Path(output).resolve()
    vendor = ROOT/'vendor'
    comp = vendor/'T2V-CompBench'
    tasks = []

    def task(name, environment, cwd, script, arguments):
        path = Path(cwd)/script
        tasks.append(dict(name=name, environment=environment, cwd=str(cwd),
                          script=str(path), script_sha256=sha256(path) if path.exists() else None,
                          argv=['{'+environment+'}', str(path)]+[str(a) for a in arguments]))

    if suite == 'vbench':
        _, audit = suites()
        task('vbench_all_16_dimensions', 'vbench_python', vendor/'VBench', 'evaluate.py',
             ['--videos_path', videos/'vbench', '--full_json_dir', vendor/'VBench/vbench/VBench_full_info.json',
              '--output_path', output/'vbench', '--mode', 'vbench_standard', '--load_ckpt_from_local', 'True',
              '--dimension']+audit['vbench']['dimensions'])
    elif suite == 'videophy2':
        for mode in ('sa', 'pc'):
            task('videophy2_'+mode, 'videophy_python', vendor/'VideoREPA/evaluation/VIDEOPHY2', 'inference.py',
                 ['--input_csv', output/'videophy2_inputs.csv', '--checkpoint', '{videophy_checkpoint}',
                  '--output_csv', output/f'videophy2_{mode}.csv', '--task', mode])
    elif suite == 't2v_compbench':
        for category, script, metadata in [
            ('consistent_attr', 'consistent_attr', 'consistent_attribute_binding'),
            ('dynamic_attr', 'dynamic_attr', 'dynamic_attribute_binding'),
            ('action_binding', 'action_binding', 'action_binding'),
            ('interaction', 'interaction', 'object_interactions')]:
            task(category, 'llava_python', comp/'LLaVA', f'llava/eval/compbench_eval_{script}.py',
                 ['--video-path', videos/f't2v_compbench/{category}', '--output-path', output/category,
                  '--read-prompt-file', comp/f'meta_data/{metadata}.json', '--t2v-model', 'wan_said',
                  '--model-path', '{llava_checkpoint}'])
        task('spatial_relationships', 'detector_python', comp,
             'Grounded-Segment-Anything/compbench_eval_spatial_relationships.py',
             ['--video-path', videos/'t2v_compbench/spatial_relationships', '--depth_folder', output/'depth',
              '--output-path', output/'spatial_relationships', '--read-prompt-file', comp/'meta_data/spatial_relationships.json',
              '--t2v-model', 'wan_said', '--output_dir', output/'spatial'])
        task('generative_numeracy', 'detector_python', comp,
             'Grounded-Segment-Anything/GroundingDINO/demo/compbench_eval_numeracy.py',
             ['--video-path', videos/'t2v_compbench/generative_numeracy', '--output-path', output/'generative_numeracy',
              '--read-prompt-file', comp/'meta_data/generative_numeracy.json', '--t2v-model', 'wan_said',
              '--output_dir', output/'numeracy'])
        task('motion_segmentation', 'detector_python', comp,
             'Grounded-Segment-Anything/compbench_motion_binding_seg.py',
             ['--video-path', videos/'t2v_compbench/motion_binding', '--read-prompt-file', comp/'meta_data/motion_binding.json',
              '--t2v-model', 'wan_said', '--total_frame', '16', '--fps', '8', '--output_dir', output/'motion_masks'])
        task('motion_tracking', 'dot_python', comp/'dot', 'compbench_eval_motion_binding.py',
             ['--video-path', videos/'t2v_compbench/video_standard/motion_binding', '--mask_folder', output/'motion_masks',
              '--read-prompt-file', comp/'meta_data/motion_binding.json', '--t2v_model', 'wan_said',
              '--output_path', output/'motion_binding', '--output_dir', output/'motion_tracks'])
    else:
        raise ValueError(suite)
    return tasks


def run(suite, jobs_path, videos, output, environments=None, plan_only=False, accept_videophy_590=False):
    jobs = read_jsonl(jobs_path)
    if not jobs or any(j['suite'] != suite for j in jobs):
        raise ValueError('Scorer jobs do not belong to the requested benchmark')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = recipes(suite, videos, output)
    status = dict(suite=suite, expected_videos=len(jobs), denominator=len(jobs), tasks=tasks,
                  official_scorers_executed=False, scores=None, proxy_metrics_used=False,
                  videophy_344_protocol_verified=False)
    write_json(output/'score_plan.json', status)
    if plan_only:
        return status
    validate_official_plan(jobs)
    if suite == 'videophy2' and not accept_videophy_590:
        raise ValueError('Pinned VideoREPA CSV has 590 rows, not 344; explicitly approve the named 590 protocol first')
    require_complete(jobs, videos)
    generation = json.loads((Path(videos)/'generation_lock.json').read_text())
    if (generation['smoke'] or generation['jobs_sha256'] != sha256(jobs_path) or
            generation['protocol'] != native_generation_protocol()):
        raise ValueError('Smoke or mismatched generation cannot be scored as a complete official benchmark')
    for job in jobs:
        path = Path(videos)/job['relative_path']
        saved = json.loads((Path(videos)/'records'/(job['job_id']+'.json')).read_text())
        if saved['job'] != job or saved['sha256'] != sha256(path):
            raise ValueError('Generation prompt mapping or content checksum changed')
        verify_video(path, generation['protocol'])
    if environments is None:
        raise ValueError('Official scorer environments and local checkpoint hash locks are required')
    env = json.loads(Path(environments).read_text())
    if not env.get('weights'):
        raise ValueError('No scorer weights were locked; refusing unrecorded scoring model downloads')
    for item in env['weights']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Official scoring checkpoint hash mismatch')
    status['weight_lock_sha256'] = sha256(environments)
    for task in tasks:
        if not task['script_sha256']:
            raise FileNotFoundError(f'Official scorer entry point is missing: {task["script"]}')
        name = {'vbench': 'VBench', 'videophy2': 'VideoREPA', 't2v_compbench': 'T2V-CompBench'}[suite]
        source_hash(check_vendor(name), Path(task['script']))
        task['argv'] = [value.format_map(env) for value in task['argv']]
        if not Path(task['argv'][0]).is_file():
            raise FileNotFoundError(f'Official scorer interpreter is not installed: {task["argv"][0]}')
    if suite == 'videophy2':
        with (output/'videophy2_inputs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['videopath', 'caption'])
            writer.writeheader()
            writer.writerows(dict(videopath=str((Path(videos)/j['relative_path']).resolve()),
                                  caption=j['original_prompt']) for j in jobs)
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for task in tasks:
        with (output/(task['name']+'.log')).open('w') as log:
            process = subprocess.run(task['argv'], cwd=task['cwd'], env=environment,
                                     stdout=log, stderr=subprocess.STDOUT)
        task['returncode'] = process.returncode
        write_json(output/'score_status.json', status)
        if process.returncode:
            raise RuntimeError(f'Official scorer {task["name"]} failed; inspect its preserved log')
    status.update(official_scorers_executed=True, raw_results_directory=str(output),
                  aggregation='Use pinned official per-category/dimension/SA/PC results; no invented cross-benchmark mean',
                  main_table_ready=False)
    write_json(output/'score_status.json', status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', choices=['t2v_compbench', 'vbench', 'videophy2'], required=True)
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--videos', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--environments')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--accept-videophy-590', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.suite, args.jobs, args.videos, args.output, args.environments,
                         args.plan_only, args.accept_videophy_590), indent=2))
