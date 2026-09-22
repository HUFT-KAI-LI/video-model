"""Frozen official LongLive loading, RNG capture, and continuous cached VAE decode."""
import importlib.metadata
import platform
import sys
from pathlib import Path
from common import ROOT, digest

EXPECTED = {
    'longlive_base.pt': '10a2aa8fcf89c77d9033f4c117405412a690e289625766619d293f0c5a208ee7',
    'lora.pt': 'c4e43b87d62d4b0614b496773639f1ab170a7ee486dc23407901e9d3a5ebc07a',
}


def provenance(config):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for real LongLive generation')
    weights = {}
    for name, expected in EXPECTED.items():
        matches = list((ROOT / config['model']['longlive']).rglob(name))
        if len(matches) != 1 or digest(matches[0]) != expected:
            raise ValueError(f'Missing or nonofficial LongLive weight: {name}')
        weights[name] = expected
    wan = ROOT / config['model']['wan']
    if not wan.is_dir():
        raise FileNotFoundError(wan)
    for path in sorted(wan.rglob('*')):
        if path.is_file() and path.suffix in ('.pth', '.safetensors', '.json') and '.cache' not in path.parts:
            weights['wan/' + str(path.relative_to(wan))] = digest(path)
    if not any(k.endswith('.safetensors') for k in weights) or not any(k.endswith('Wan2.1_VAE.pth') for k in weights):
        raise ValueError('Wan transformer/VAE assets incomplete')
    sources = {str(p.relative_to(ROOT)): digest(p)
               for p in sorted((ROOT / 'restream_mvp/code/LongLive').rglob('*'))
               if p.is_file() and p.suffix in ('.py', '.yaml')}
    for folder in (ROOT / 'experiments/p0a_failure_mining', ROOT / 'restream_mvp/restream'):
        sources.update({str(p.relative_to(ROOT)): digest(p) for p in sorted(folder.glob('*.py'))})
    return dict(weights=weights, source_sha256=sources, torch=str(torch.__version__),
                cuda=torch.version.cuda, device=torch.cuda.get_device_name(0),
                python=platform.python_version(), cuda_device_count=torch.cuda.device_count(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                packages={p: importlib.metadata.version(p) for p in
                          ('transformers', 'peft', 'diffusers', 'omegaconf', 'av')})


def load(config):
    import torch
    sys.path.insert(0, str(ROOT / 'restream_mvp'))
    from restream.runtime import load_pipeline
    model = dict(config['model'])
    for key in ('wan', 'longlive'):
        model[key] = str((ROOT / model[key]).resolve())
    model['component_devices'] = {'text_encoder': 'cpu', 'vae': 'cpu'}
    pipe = load_pipeline({'model': model, 'data': config}, torch.device('cuda'))
    from utils.memory import DynamicSwapInstaller
    DynamicSwapInstaller.install_model(pipe.text_encoder, device=torch.device('cuda'))
    return pipe


def rng_state():
    import torch
    return {'cpu': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def set_rng(state):
    import torch
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state_all(state['cuda'])


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return value


def decode_video(pipe, latents, path, num_frames, fps):
    """Keep VAE temporal cache across chunks; write frames without retaining full RGB video."""
    import av
    import torch
    pipe.generator.to('cpu')
    pipe.kv_cache1 = None
    pipe.crossattn_cache = None
    torch.cuda.empty_cache()
    pipe.vae.to('cuda')
    pipe.vae.model.clear_cache()
    count = 0
    try:
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('libx264', rate=fps)
            stream.width, stream.height = latents.shape[-1] * 8, latents.shape[-2] * 8
            stream.pix_fmt = 'yuv420p'
            stream.options = {'crf': '18'}
            for start in range(0, latents.shape[1], 3):
                pixels = pipe.vae.decode_to_pixel(latents[:, start:start + 3].to('cuda'), use_cache=True)
                expected = 9 if start == 0 else 12
                if pixels.shape[1] != expected:
                    raise RuntimeError(f'Unexpected VAE temporal shape: {pixels.shape}')
                arrays = ((pixels[0] * .5 + .5).clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
                for array in arrays:
                    if count >= num_frames:
                        break
                    frame = av.VideoFrame.from_ndarray(array, format='rgb24')
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    count += 1
            for packet in stream.encode():
                container.mux(packet)
    finally:
        pipe.vae.model.clear_cache()
        pipe.vae.to('cpu')
        torch.cuda.empty_cache()
        pipe.generator.to('cuda')
    if count != num_frames:
        raise RuntimeError(f'Wrote {count} frames; expected {num_frames}')
    return count
