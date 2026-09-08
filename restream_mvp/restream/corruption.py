import torch


def corrupt_history(history, generator, probability=0.8, sigma_min=0.02, sigma_max=0.12):
    """Explicit generator makes corruption identical across A/B/C variants."""
    result = history.clone()
    if torch.rand((), generator=generator, device=history.device).item() >= probability:
        return result
    sigma = sigma_min + (sigma_max - sigma_min) * torch.rand((), generator=generator, device=history.device)
    noise = torch.randn(history.shape, generator=generator, device=history.device, dtype=history.dtype)
    result = result + sigma * noise
    if torch.rand((), generator=generator, device=history.device).item() < 0.5:
        result = torch.roll(result, 1, dims=-1)
    return result
