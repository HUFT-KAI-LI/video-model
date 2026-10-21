"""Download the public official VideoPhy-2 evaluator via HF mirror, with a weight lock."""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from ..io_utils import sha256, write_json

VIDEOPHY_REPO = 'videophysics/videophy_2_auto'
VIDEOPHY_REVISION = 'ca38370e1dcc79c99c7017dd8755d0a087299afc'


def download(output, interpreter):
    endpoint = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
    api = HfApi(endpoint=endpoint)
    info = api.model_info(VIDEOPHY_REPO, revision=VIDEOPHY_REVISION, files_metadata=True)
    files = [s for s in info.siblings if s.rfilename.endswith(('.bin', '.json', '.model'))]
    total = sum(s.size for s in files)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    weights = output/'videophy2_auto'
    already = sum((weights/s.rfilename).stat().st_size for s in files if (weights/s.rfilename).is_file())
    if shutil.disk_usage(output).free < max(0, total-already)+50*2**30:
        raise OSError('Insufficient space for evaluator plus 50 GiB reserve; no old data deleted')
    plan = dict(repo=VIDEOPHY_REPO, revision=VIDEOPHY_REVISION, endpoint=endpoint,
                files=[dict(name=s.rfilename, bytes=s.size, lfs_sha256=s.lfs.sha256 if s.lfs else None)
                       for s in files], total_bytes=total, max_workers=2, feature_cache_bytes=0)
    write_json(output/'download_plan.json', plan)
    print(plan, flush=True)
    snapshot_download(VIDEOPHY_REPO, revision=VIDEOPHY_REVISION, endpoint=endpoint,
                      local_dir=weights, allow_patterns=[s.rfilename for s in files], max_workers=2)
    locked = []
    for item in files:
        path = weights/item.rfilename
        actual = sha256(path)
        if path.stat().st_size != item.size or (item.lfs and actual != item.lfs.sha256):
            raise ValueError(f'Downloaded evaluator file failed size/SHA256: {item.rfilename}')
        locked.append(dict(path=str(path), sha256=actual, bytes=item.size))
    subprocess.run([str(interpreter), '-c',
                    'import transformers, peft, decord; from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration'],
                   check=True, cwd=Path(__file__).resolve().parents[1]/'vendor/VideoREPA/evaluation/VIDEOPHY2')
    write_json(output/'videophy2_environment.json', dict(videophy_python=str(Path(interpreter).absolute()),
               videophy_checkpoint=str(weights), weights=locked, checkpoint_source=plan,
               integration_status='weights verified and imports checked; real video scoring still required'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--interpreter', required=True)
    args = parser.parse_args()
    download(args.output, args.interpreter)
