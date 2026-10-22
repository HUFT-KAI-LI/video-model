"""Real VideoPhy-2 transport check on one video; never a benchmark summary."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

from ..io_utils import sha256, write_json
from .benchmarks import check_vendor, source_hash
from .inference import native_generation_protocol, verify_video
from .score import recipes


def probe(record, environment, output):
    saved = json.loads(Path(record).read_text())
    video = Path(saved['path'])
    if saved['sha256'] != sha256(video):
        raise ValueError('Probe video checksum changed')
    verify_video(video, native_generation_protocol())
    env = json.loads(Path(environment).read_text())
    for weight in env['weights']:
        if sha256(weight['path']) != weight['sha256']:
            raise ValueError('Probe scoring model checksum changed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'videophy2_inputs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['videopath', 'caption'])
        writer.writeheader()
        writer.writerow(dict(videopath=str(video), caption=saved['job']['original_prompt']))
    status = dict(integration_probe_only=True, formal_benchmark_complete=False,
                  benchmark_denominator=590, probe_videos=1, scores_are_benchmark_results=False,
                  video_sha256=saved['sha256'], checkpoint_environment_sha256=sha256(environment), tasks=[])
    write_json(output/'probe_status.json', status)
    for task in recipes('videophy2', video.parent.parent, output):
        source_hash(check_vendor('VideoREPA'), Path(task['script']))
        task['argv'] = [value.format_map(env) for value in task['argv']]
        with (output/(task['name']+'.log')).open('w') as log:
            process = subprocess.run(task['argv'], cwd=task['cwd'], stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
        task['returncode'] = process.returncode
        status['tasks'].append(task)
        write_json(output/'probe_status.json', status)
        if process.returncode:
            raise RuntimeError('Real official scorer integration probe failed; inspect its log')
    for mode in ('sa', 'pc'):
        with (output/f'videophy2_{mode}.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        if (len(rows) != 1 or rows[0]['videopath'] != str(video) or
                rows[0]['caption'] != saved['job']['original_prompt'] or
                float(rows[0]['score']) not in (1, 2, 3, 4, 5)):
            raise ValueError('Official scorer returned a missing, unparseable, or mismatched score')
    status['integration_succeeded'] = True
    write_json(output/'probe_status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    probe(args.record, args.environment, args.output)
