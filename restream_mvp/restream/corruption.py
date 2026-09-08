import torch


def corrupt_history(history, generator, probability=0.8, sigma_min=0.02, sigma_max=0.12, protected_prefix=3):
    """Explicit generator makes corruption identical across A/B/C variants."""
    result = history.clone()
    if history.shape[1] <= protected_prefix:
        return result
    if torch.rand((), generator=generator, device=history.device).item() >= probability:
        return result
    sigma = sigma_min + (sigma_max - sigma_min) * torch.rand((), generator=generator, device=history.device)
    noise = torch.randn(history[:, protected_prefix:].shape, generator=generator, device=history.device, dtype=history.dtype)
    result[:, protected_prefix:] = result[:, protected_prefix:] + sigma * noise
    if torch.rand((), generator=generator, device=history.device).item() < 0.5:
        result[:, protected_prefix:] = torch.roll(result[:, protected_prefix:], 1, dims=-1)
    return result
