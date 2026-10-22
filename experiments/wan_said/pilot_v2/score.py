"""Run pinned official scoring entry points without changing the expected denominator."""
import argparse
import contextlib
import csv
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from ..io_utils import read_jsonl, sha256, write_json
from ..runtime import ROOT
from .benchmarks import check_vendor, require_complete, source_hash, suites, validate_official_plan
from .inference import native_generation_protocol, verify_video
from .aggregate import save_summary, vbench, videophy2


def execute_tasks(tasks, output, environment, devices, persist):
    if not devices:
        for task in tasks:
            with (output/(task['name']+'.log')).open('w') as log:
                process = subprocess.run(task['argv'], cwd=task['cwd'], env=environment,
                                         stdout=log, stderr=subprocess.STDOUT)
            task['returncode'] = process.returncode
            persist()
            if process.returncode:
                raise RuntimeError(f'Official scorer {task["name"]} failed; inspect its preserved log')
        return
    if len(set(devices)) != len(devices):
        raise ValueError('Each parallel scorer must have a distinct physical GPU')
    pending, active = list(tasks), []
    with contextlib.ExitStack() as stack:
        try:
            while pending or active:
                used = {device for _, _, device in active}
                for device in devices:
                    if not pending or device in used:
                        continue
                    task = pending.pop(0)
                    log = stack.enter_context((output/(task['name']+'.log')).open('w'))
                    task['cuda_visible_devices'] = device
                    process = subprocess.Popen(task['argv'], cwd=task['cwd'],
                        env=dict(environment, CUDA_VISIBLE_DEVICES=device), stdout=log, stderr=subprocess.STDOUT)
                    active.append((process, task, device))
                for process, task, device in active[:]:
                    code = process.poll()
                    if code is None:
                        continue
                    task['returncode'] = code
                    active.remove((process, task, device))
                    persist()
                    if code:
                        raise RuntimeError(f'Official scorer {task["name"]} failed; inspect its preserved log')
                if active:
                    time.sleep(2)
        finally:
            for process, _, _ in active:
                if process.poll() is None:
                    process.terminate()
            for process, _, _ in active:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


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


def run(suite, jobs_path, videos, output, environments=None, plan_only=False,
        accept_videophy_590=False, aggregate_only=False, subset_lock=None, score_devices=None):
    full_jobs = read_jsonl(jobs_path)
    jobs = full_jobs
    if subset_lock:
        if suite not in ('videophy2', 'vbench'):
            raise ValueError('Unsupported subset benchmark')
        from .subset import select
        jobs = select(full_jobs, jobs_path, subset_lock)
    if not jobs or any(j['suite'] != suite for j in jobs):
        raise ValueError('Scorer jobs do not belong to the requested benchmark')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = recipes(suite, videos, output)
    metadata = None
    if subset_lock and suite == 'vbench':
        from .subset import vbench_metadata
        metadata = vbench_metadata(jobs)
        metadata_path = output/'vbench_subset_metadata.json'
        write_json(metadata_path, metadata)
        template = tasks[0]
        dimensions = template['argv'][template['argv'].index('--dimension')+1:]
        tasks = []
        for dimension in dimensions:
            task = dict(template, name='vbench_'+dimension)
            argv = template['argv'][:template['argv'].index('--dimension')]+['--dimension', dimension]
            argv[argv.index('--full_json_dir')+1] = str(metadata_path)
            argv[argv.index('--output_path')+1] = str(output/'vbench'/dimension)
            task['argv'] = argv
            tasks.append(task)
    status = dict(suite=suite, expected_videos=len(jobs), denominator=len(jobs), tasks=tasks,
                  official_scorers_executed=False, scores=None, proxy_metrics_used=False,
                  benchmark_version='VideoPhy-2' if suite == 'videophy2' else suite,
                  jobs_sha256=sha256(jobs_path), main_table_ready=False)
    selected_sha = sha256(subset_lock) if subset_lock else None
    status['subset_lock_sha256'] = selected_sha
    if subset_lock:
        status.update(evaluation_scope='subset_pilot', subset_lock_sha256=selected_sha,
                      full_benchmark_denominator=len(full_jobs), formal_benchmark_complete=False,
                      subset_table_ready=False)
        if suite == 'vbench':
            status.update(prompt_count=len({j['prompt_id'] for j in jobs}), samples_per_prompt=5,
                          subset_metadata_sha256=sha256(metadata_path))
    write_json(output/'score_plan.json', status)
    if plan_only:
        return status
    previous = json.loads((output/'score_status.json').read_text()) if aggregate_only else None
    (output/'summary.csv').unlink(missing_ok=True)
    (output/'summary.json').unlink(missing_ok=True)
    write_json(output/'score_status.json', status)
    validate_official_plan(full_jobs)
    require_complete(jobs, videos)
    generation = json.loads((Path(videos)/'generation_lock.json').read_text())
    if generation.get('subset_lock_sha256') != selected_sha:
        raise ValueError('Subset generation cannot be relabeled as a full benchmark or another subset')
    if (generation['smoke'] or generation['jobs_sha256'] != sha256(jobs_path) or
            generation['protocol'] != native_generation_protocol()):
        raise ValueError('Smoke or mismatched generation cannot be scored as a complete official benchmark')
    status['generation_lock_sha256'] = sha256(Path(videos)/'generation_lock.json')
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
    if aggregate_only:
        for key in ('suite', 'jobs_sha256', 'generation_lock_sha256', 'weight_lock_sha256', 'subset_lock_sha256'):
            if previous.get(key) != status[key]:
                raise ValueError(f'Cannot reuse official scores from another {key}')
        if (not previous['official_scorers_executed'] or
                len(previous['tasks']) != len(tasks) or
                any(old.get('returncode') != 0 or old['script_sha256'] != new['script_sha256'] or
                    old['argv'] != new['argv'] for old, new in zip(previous['tasks'], tasks))):
            raise ValueError('Aggregation requires all pinned official scoring tasks to have succeeded')
        status['tasks'] = previous['tasks']
    else:
        environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        if suite == 'vbench' and (output/'vbench').exists():
            archive = output/'prior_attempts'/uuid.uuid4().hex
            archive.mkdir(parents=True)
            shutil.move(output/'vbench', archive/'vbench')
            for path in output.glob('vbench_*.log'):
                shutil.move(path, archive/path.name)
        execute_tasks(tasks, output, environment, score_devices,
                      lambda: write_json(output/'score_status.json', status))
    status.update(official_scorers_executed=True, raw_results_directory=str(output),
                  main_table_ready=False)
    write_json(output/'score_status.json', status)
    try:
        paths = ([output/'videophy2_sa.csv', output/'videophy2_pc.csv'] if suite == 'videophy2' else
                 sorted((output/'vbench').rglob('*_eval_results.json')) if suite == 'vbench' else [])
        raw_hashes = {str(p.relative_to(output)): sha256(p) for p in paths}
        if aggregate_only and previous.get('raw_result_sha256') != raw_hashes:
            raise ValueError('Official raw scoring results changed; rerun the evaluator, not only aggregation')
        status['raw_result_sha256'] = raw_hashes
        write_json(output/'score_status.json', status)
        if suite == 'videophy2':
            summary = videophy2(jobs, videos, output/'videophy2_sa.csv', output/'videophy2_pc.csv')
        elif suite == 'vbench':
            summary = vbench(jobs, videos, paths, metadata=metadata,
                             allow_official_color_exclusions=bool(subset_lock))
        else:
            status['aggregation'] = 'T2V-CompBench category aggregation is not implemented; raw results only'
            write_json(output/'score_status.json', status)
            return status
        summary.update(jobs_sha256=status['jobs_sha256'], generation=generation,
                       weight_lock_sha256=status['weight_lock_sha256'],
                       generation_lock_sha256=status['generation_lock_sha256'],
                       exploratory=bool(generation.get('exploratory')),
                       publication_eligible=not generation.get('exploratory', False))
        if subset_lock:
            summary.update(evaluation_scope='subset_pilot', subset_lock_sha256=selected_sha,
                           full_benchmark_denominator=len(full_jobs), formal_benchmark_complete=False,
                           publication_eligible=False, main_table_ready=False,
                           subset_table_ready=summary['complete'])
            if suite == 'vbench':
                summary.update(prompt_count=status['prompt_count'], samples_per_prompt=5,
                               subset_metadata_sha256=status['subset_metadata_sha256'])
        save_summary(summary, output, generation['group'])
        status.update(scores=summary['scores'], main_table_ready=summary['main_table_ready'],
                      summary_sha256=sha256(output/'summary.json'))
        if subset_lock:
            status['subset_table_ready'] = summary['subset_table_ready']
    except Exception as error:
        status.update(main_table_ready=False, aggregation_error=repr(error))
        write_json(output/'score_status.json', status)
        raise
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
    parser.add_argument('--accept-videophy-590', action='store_true', help='Deprecated compatibility flag; VideoPhy-2 is 590 prompts')
    parser.add_argument('--aggregate-only', action='store_true', help='Recheck and aggregate an already successful official scoring run')
    parser.add_argument('--subset-lock', help='Explicit pilot subset; cannot produce a full benchmark table')
    parser.add_argument('--score-devices', nargs='+', help='Physical CUDA indices for parallel official scorer tasks')
    args = parser.parse_args()
    print(json.dumps(run(args.suite, args.jobs, args.videos, args.output, args.environments,
                         args.plan_only, args.accept_videophy_590, args.aggregate_only, args.subset_lock,
                         args.score_devices), indent=2))
