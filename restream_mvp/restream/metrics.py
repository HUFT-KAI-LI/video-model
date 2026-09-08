import torch


def future_errors(predicted, target, anchor_index, window_sec):
    if predicted.shape != target.shape or anchor_index >= target.shape[1] - 1:
        raise ValueError("Mismatched tensors or missing future")
    result = {}
    dt = window_sec / (target.shape[1] - 1)
    for seconds in (.5, 1., 2.):
        count = min(int(seconds / dt + 1e-6), target.shape[1] - anchor_index - 1)
        key = f"future_latent_mse_{seconds:g}s"
        if count == 0:
            result[key] = None
        else:
            error = predicted[:, anchor_index + 1:anchor_index + 1 + count].float() - target[:, anchor_index + 1:anchor_index + 1 + count].float()
            result[key] = error.square().mean().item()
    result["future_latent_mse"] = (predicted[:, anchor_index + 1:].float() - target[:, anchor_index + 1:].float()).square().mean().item()
    return result
