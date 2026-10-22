"""Prepare isolated official VBench dependencies and hash-locked local model files."""
import argparse
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time
import zipfile

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import requests

from ..io_utils import sha256, write_json
from ..runtime import ROOT


def prepare(output, environment):
    output, environment = Path(output).resolve(), Path(environment).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ready = output/'vbench_environment.json'
    if ready.exists():
        raise FileExistsError('A verified VBench environment already exists; do not overwrite its lock')
    cache, hfhome = output/'vbench_cache', output/'hf_home'
    sources = []
    runtime = dict(os.environ, VBENCH_CACHE_DIR=str(cache), HF_HOME=str(hfhome),
                   HF_ENDPOINT='https://hf-mirror.com', HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0',
                   MAX_JOBS='4', CUDA_VISIBLE_DEVICES='', FORCE_CUDA='0')
    # CPU-only extension compilation does not compete with the generation workers.
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)], check=True)
    python = environment/'bin/python'
    constraints = output/'dependency_constraints.txt'
    constraints.write_text('torch==2.5.1+cu124\ntorchvision==0.20.1+cu124\nnumpy==1.26.4\n'
                           'transformers==4.33.2\ntimm==0.9.16\nopencv-python==4.10.0.84\n')
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', '-c', str(constraints),
                    '-r', str(ROOT/'vendor/VBench/requirements.txt'), 'pyiqa==0.1.10',
                    'setuptools<81', 'ninja', 'hydra-core>=1.1'], env=runtime, check=True)

    def clone(name, url, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        lock = output/(name+'_source.json')
        if not lock.exists():
            repo = url.removeprefix('https://github.com/').removesuffix('.git')
            response = requests.get(f'https://api.github.com/repos/{repo}/commits/HEAD', timeout=30)
            response.raise_for_status()
            revision = response.json()['sha']
            archive_url = f'https://codeload.github.com/{repo}/zip/{revision}'
            response = requests.get(archive_url, timeout=(30, 120))
            response.raise_for_status()
            archive = output/(name+'_source.zip')
            archive.write_bytes(response.content)
            staging = output/(name+'_source_unpack')
            staging.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as zipped:
                for item in zipped.infolist():
                    if not (staging/item.filename).resolve().is_relative_to(staging.resolve()):
                        raise ValueError('Unsafe path in official source archive')
                zipped.extractall(staging)
            unpacked = next(staging.iterdir())
            unpacked.rename(destination)
            staging.rmdir()
            write_json(lock, dict(name=name, url=url, revision=revision, archive_url=archive_url,
                                  archive_sha256=sha256(archive)))
        sources.append(json.loads(lock.read_text()))

    detectron = output/'detectron2'
    clone('detectron2', 'https://github.com/facebookresearch/detectron2.git', detectron)
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-cache-dir', '--no-build-isolation',
                    '--no-deps', str(detectron)], env=runtime, check=True)
    clone('dino', 'https://github.com/facebookresearch/dino.git', cache/'dino_model/facebookresearch_dino_main')
    vocab = cache/'ViCLIP/bpe_simple_vocab_16e6.txt.gz'
    vocab.parent.mkdir(parents=True, exist_ok=True)
    locate_vocab = ('import importlib.util; from pathlib import Path; '
                    'print(Path(importlib.util.find_spec("clip").origin).parent/"bpe_simple_vocab_16e6.txt.gz")')
    source_vocab = Path(subprocess.check_output([str(python), '-c', locate_vocab], text=True).strip())
    shutil.copy2(source_vocab, vocab)
    sources.append(dict(name='OpenAI CLIP BPE vocabulary', installed_package='openai-clip', sha256=sha256(vocab)))
    api = HfApi(endpoint=runtime['HF_ENDPOINT'])

    def hf(repo, filename, destination, repo_type='model', revision=None, expected_sha256=None):
        info = api.repo_info(repo, repo_type=repo_type, files_metadata=True, revision=revision, timeout=60)
        entry = next(s for s in info.siblings if s.rfilename == filename)
        if expected_sha256 and (not entry.lfs or entry.lfs.sha256 != expected_sha256):
            raise ValueError('Mirrored weight metadata differs from the pinned content digest')
        destination.parent.mkdir(parents=True, exist_ok=True)
        print('weight', repo, info.sha, filename, entry.size, flush=True)
        if not (destination.exists() and entry.lfs and sha256(destination) == entry.lfs.sha256):
            source = hf_hub_download(repo, filename, repo_type=repo_type, revision=info.sha,
                                     endpoint=runtime['HF_ENDPOINT'], local_dir=destination.parent)
            if Path(source) != destination:
                shutil.move(source, destination)
        if entry.lfs and sha256(destination) != entry.lfs.sha256:
            raise ValueError(f'Invalid mirrored official weight {filename}')
        sources.append(dict(repo=repo, filename=filename, revision=info.sha,
                            upstream_sha256=entry.lfs.sha256 if entry.lfs else None))
        print('verified', destination, sha256(destination), flush=True)

    used = 'OpenGVLab/VBench_Used_Models'
    for filename, folder in [('l16_ptk710_ftk710_ftk400_f16_res224.pth', 'umt_model'),
                              ('grit_b_densecap_objectdet.pth', 'grit_model'),
                              ('ViClip-InternVid-10M-FLT.pth', 'ViCLIP')]:
        hf(used, filename, cache/folder/filename)
    hf('lalala125/AMT', 'amt-s.pth', cache/'amt_model/amt-s.pth')
    hf('xinyu1205/recognize-anything', 'tag2text_swin_14m.pth',
       cache/'caption_model/tag2text_swin_14m.pth', 'space')

    def url_download(url, destination, expected=None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix+'.partial')
            for attempt in range(5):
                try:
                    request_url = ('https://ghproxy.net/'+url if attempt and url.startswith(
                        ('https://github.com/', 'https://raw.githubusercontent.com/')) else url)
                    with requests.get(request_url, stream=True, timeout=(30, 90)) as response:
                        response.raise_for_status()
                        with temporary.open('wb') as stream:
                            for chunk in response.iter_content(8*1024**2):
                                if shutil.disk_usage(output).free < 50*1024**3:
                                    raise OSError('VBench download stopped to preserve 50 GiB free disk')
                                stream.write(chunk)
                    temporary.replace(destination)
                    break
                except (requests.RequestException, OSError):
                    if attempt == 4:
                        raise
                    time.sleep(15*(attempt+1))
        if expected and sha256(destination) != expected:
            raise ValueError(f'Official URL SHA mismatch: {url}')
        sources.append(dict(url=url, path=str(destination), sha256=sha256(destination)))

    for name, checksum in [('ViT-B-32.pt', '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'),
                           ('ViT-L-14.pt', 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836')]:
        url_download(f'https://openaipublic.azureedge.net/clip/models/{checksum}/{name}', cache/'clip_model'/name, checksum)
    url_download('https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
                 cache/'dino_model/dino_vitbase16_pretrain.pth')
    url_download('https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth',
                 cache/'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth')
    url_download('https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth',
                 cache/'aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth')
    raft = cache/'raft_model/models/raft-things.pth'
    # Publisher's Dropbox URL is unreachable here. Two independent HF copies
    # agree on this digest; record that evidence, not a nonexistent publisher hash.
    raft_sha = 'fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1'
    hf('dc-ai/vbench_pretrained_models', 'raft_model/models/raft-things.pth', raft,
       revision='d6ddad44430e861088ae49409dac073d04dc7e12', expected_sha256=raft_sha)
    sources[-1].update(original_publisher_url='https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip',
                       publisher_checksum_available=False,
                       mirror_corroboration=dict(repo='sbalani/raft-things',
                       revision='3cda7e7fe51ad8f594c2824a6d6ee13b7f61afdc', sha256=raft_sha))
    bert = api.model_info('bert-base-uncased')
    bert_cache = hfhome/'hub'
    snapshot_download('bert-base-uncased', revision=bert.sha, endpoint=runtime['HF_ENDPOINT'],
                      cache_dir=bert_cache, allow_patterns=['config.json', 'pytorch_model.bin',
                      'tokenizer.json', 'tokenizer_config.json', 'vocab.txt'], max_workers=2)
    ref = bert_cache/'models--bert-base-uncased/refs/main'
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(bert.sha)
    sources.append(dict(repo='bert-base-uncased', revision=bert.sha))
    # The official entry point inherits these locations without changing its source.
    wrapper = output/'vbench_python'
    wrapper.write_text('#!/bin/sh\nexport VBENCH_CACHE_DIR='+shlex.quote(str(cache))+
                       '\nexport HF_HOME='+shlex.quote(str(hfhome))+'\nexec '+shlex.quote(str(python))+' "$@"\n')
    wrapper.chmod(0o755)
    # Import every evaluator before publishing readiness; no GPU memory is allocated.
    from .benchmarks import suites
    _, audit = suites()
    probe = 'import importlib; '+ '; '.join('importlib.import_module('+repr('vbench.'+d)+')'
                                           for d in audit['vbench']['dimensions'])
    subprocess.run([str(wrapper), '-c', probe], cwd=ROOT/'vendor/VBench', env=runtime, check=True)
    freeze = subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True)
    (output/'vbench_pip_freeze.txt').write_text(freeze)
    files = sorted(p for folder in (cache, hfhome) for p in folder.rglob('*')
                   if p.is_file() and '.git' not in p.parts and '.cache' not in p.parts)
    files.extend([wrapper, output/'vbench_pip_freeze.txt'])
    weights = [dict(path=str(p), sha256=sha256(p), bytes=p.stat().st_size) for p in files]
    write_json(ready, dict(vbench_python=str(wrapper), weights=weights, checkpoint_sources=sources,
                          runtime_environment=dict(VBENCH_CACHE_DIR=str(cache), HF_HOME=str(hfhome)),
                          integration_status='weights locked and all evaluator imports checked; real video scoring pending'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--environment', required=True)
    args = parser.parse_args()
    prepare(args.output, args.environment)
