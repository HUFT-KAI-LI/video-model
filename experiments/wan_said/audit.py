import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch

from .cache_features import load_features
from .data import load_split
from .io_utils import read_jsonl, sha256, write_json
from .runtime import ROOT, config


def audit():
    cfg = config()
    output = ROOT / 'reports'
    output.mkdir(exist_ok=True)
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,driver_version',
                                   '--format=csv,noheader'], text=True)
    packages = {p: importlib.metadata.version(p) for p in
                ('torch', 'torchvision', 'transformers', 'diffusers', 'peft', 'av', 'timm', 'remotezip',
                 'imageio', 'imageio-ffmpeg', 'safetensors', 'huggingface-hub')}
    weights = {name: dict(path=str(Path(cfg['wan_weights']) / name),
                         bytes=(Path(cfg['wan_weights']) / name).stat().st_size,
                         sha256=sha256(Path(cfg['wan_weights']) / name)) for name in
               ('diffusion_pytorch_model.safetensors', 'models_t5_umt5-xxl-enc-bf16.pth', 'Wan2.1_VAE.pth')}
    write_json(output / 'phase0.json', dict(gpus=gpu, packages=packages, weights=weights,
               cwd=str(ROOT.parents[1]), branch=subprocess.check_output(
                   ['git', '-C', str(ROOT), 'branch', '--show-current'], text=True).strip(),
               upstream=json.loads((ROOT / 'upstream_lock.json').read_text()),
               isolation='Separate worktree, own venv, read-only shared Wan weights and smoke videos; no LongLive patches',
               disk=subprocess.check_output(['df', '-h', '/data', '/data1'], text=True)))
    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                          cwd=ROOT.parents[1], text=True, capture_output=True)
    (output / 'unit_tests.txt').write_text(test.stdout + test.stderr)
    assert test.returncode == 0, test.stderr
    reports, steps, gradients = {}, {}, {}
    for group in ('A', 'B', 'C'):
        run = ROOT / 'runs' / f'smoke_{group}'
        reports[group] = json.loads((run / 'report.json').read_text())
        steps[group] = read_jsonl(run / 'steps.jsonl')
        gradients[group] = json.loads((run / 'gradient_audit.json').read_text())
        write_json(output / f'smoke_{group}.json', dict(report=reports[group], steps=steps[group], gradients=gradients[group]))
        assert reports[group]['serialization_prediction_max_error'] == 0
        assert gradients[group]['fm']['nonzero_blocks'] == list(range(1, 31))
        if group != 'A':
            assert gradients[group]['distill']['nonzero_blocks'] == list(range(1, 16))
    assert len({reports[g]['initial_lora_sha256'] for g in reports}) == 1
    for group in ('B', 'C'):
        for a, b in zip(steps['A'], steps[group]):
            assert (a['sample_id'], a['sigma'], a['timestep']) == (b['sample_id'], b['sigma'], b['timestep'])
        assert steps['A'][0]['fm'] == steps[group][0]['fm']
    sampling_matches = 0
    for split in ('train', 'val'):
        for row in load_split(cfg['manifests'], split):
            feature = load_features(cfg, row)
            assert feature['teacher']['sampling'] == feature['vae']['sampling']
            assert len(feature['teacher']['teacher_frame_indices']) == 16
            assert feature['teacher']['raw_tokens'].shape[-1] == cfg['teacher_dim']
            sampling_matches += 1
    for mode in ('full', 'said'):
        write_json(output / f'ground_{mode}.json', json.loads((ROOT / 'runs' / f'ground_{mode}' / 'report.json').read_text()))
    generation = json.loads((ROOT / 'runs/smoke_generation/generation_report.json').read_text())
    assert all(not v['teacher_used'] and not v['auxiliary_modules_used'] for v in generation['videos'])
    write_json(output / 'generation_summary.json', generation)
    selection = ROOT / 'datasets/openvid/selection.json'
    if selection.exists():
        write_json(output / 'openvid_selection.json', json.loads(selection.read_text()))
    write_json(output / 'engineering_validation.json', dict(unit_tests_passed=True,
               matched_initialization=True, matched_sample_noise_order=True,
               matched_initial_fm=True, teacher_wan_sampling_matches=sampling_matches,
               cached_splits=['train', 'val'], cached_test_count=0,
               native_generation_videos=len(generation['videos']),
               peak_allocated_gib={g: r['peak_allocated_gib'] for g, r in reports.items()},
               seconds={g: r['seconds'] for g, r in reports.items()},
               phase2='passed technical smoke only', phase3='not run',
               hypothesis='Not established: validation retrieval near chance and no generation improvement evidence',
               unsaid='Do not add before validating Said selectivity and native video quality'))
    (output / 'requirements-frozen.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    print(json.dumps(json.loads((output / 'engineering_validation.json').read_text()), indent=2))


if __name__ == '__main__':
    audit()
