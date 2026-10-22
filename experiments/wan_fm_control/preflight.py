"""Strict numerical resume checks without claiming bitwise DDP reproducibility."""
import torch


def resume_equivalence(resumed, fresh):
    for key in ('contract', 'config', 'group', 'steps', 'initial_lora_sha256'):
        if resumed[key] != fresh[key]:
            raise AssertionError(f'Resume identity changed: {key}')
    keys = ('step', 'rank', 'sample_id', 'noise_seed', 'timestep', 'sigma', 'fm', 'sampling')
    if ([{k: r[k] for k in keys} for r in resumed['records']] !=
            [{k: r[k] for k in keys} for r in fresh['records']]):
        raise AssertionError('Resume changed exact samples, preprocessing, diffusion noise or FM losses')
    if set(resumed['lora']) != set(fresh['lora']):
        raise AssertionError('Resume adapter keys differ')
    maximum = 0.
    for key, value in resumed['lora'].items():
        other = fresh['lora'][key]
        torch.testing.assert_close(value, other, rtol=1e-7, atol=1e-10)
        maximum = max(maximum, (value-other).abs().max().item())
    a, b = resumed['optimizer'], fresh['optimizer']
    if a['param_groups'] != b['param_groups'] or set(a['state']) != set(b['state']):
        raise AssertionError('Resume optimizer layout/hyperparameters changed')
    optimizer_maximum = 0.
    for index, state in a['state'].items():
        if set(state) != set(b['state'][index]):
            raise AssertionError('Resume optimizer state keys changed')
        for key, value in state.items():
            other = b['state'][index][key]
            if torch.is_tensor(value):
                torch.testing.assert_close(value, other, rtol=0. if key == 'step' else 1e-7,
                                           atol=0. if key == 'step' else 1e-12)
                optimizer_maximum = max(optimizer_maximum, (value-other).abs().max().item())
            elif value != other:
                raise AssertionError('Resume optimizer non-tensor state changed')
    return dict(resume_exact=resumed['final_lora_sha256'] == fresh['final_lora_sha256'],
        resume_numerically_equivalent=True, losses_samples_noise_exact=True,
        lora_max_abs_difference=maximum, optimizer_max_abs_difference=optimizer_maximum,
        lora_tolerance=dict(rtol=1e-7, atol=1e-10), optimizer_tolerance=dict(rtol=1e-7, atol=1e-12),
        explanation='Floating-point reduction/bucket ordering can prevent byte-identical DDP resume; both states are preserved.')
