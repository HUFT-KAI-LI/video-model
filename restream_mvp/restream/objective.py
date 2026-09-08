import torch
from .anchor_injector import HardAnchorInjector, timestamp_to_latent_index
from .corruption import corrupt_history


def prepare(pipeline, batch, device, config, rng, force_drift=False):
    pixels = batch["pixels"].to(device, dtype=torch.bfloat16)
    # MVP deliberately uses per-GPU batch=1, allowing independently jittered anchors.
    if pixels.shape[0] != 1:
        raise ValueError("MVP expects per-GPU batch size 1")
    index, actual_time = timestamp_to_latent_index(float(batch["anchor_sec"][0]),
                                                  float(batch["window_sec"][0]), pixels.shape[2])
    with torch.no_grad():
        gt = pipeline.vae.encode_to_latent(pixels).to(dtype=torch.bfloat16)
        if gt.shape[1] != (pixels.shape[2] - 1) // 4 + 1:
            raise ValueError("VAE temporal compression does not match the verified mapping")
        real = HardAnchorInjector(pipeline.vae).encode_anchor(pixels[:, :, 4 * index]).to(gt.dtype)
        conditioning = pipeline.text_encoder(batch["caption"])
        history = corrupt_history(gt[:, :index + 1], rng,
                                  probability=1 if force_drift else config["reanchor"]["drift_probability"])
    return gt, history, real, conditioning, index, actual_time


def future_loss(pipeline, adapter, gt, history, real, conditioning, index, rng, regularization):
    from utils.loss import get_denoising_loss
    corrected = adapter(history[:, -1:], real)
    clean_context = torch.cat((history[:, :-1], corrected, gt[:, index + 1:]), dim=1)
    # Teacher forcing computes differentiable clean-context K/V in the same graph.
    # The real-image latent only affects FUTURE blocks; same-block losses excluded.
    count = gt.shape[1] // pipeline.num_frame_per_block
    ids = torch.randint(0, 1000, (gt.shape[0], count), generator=rng, device=gt.device)
    times = pipeline.scheduler.timesteps.to(gt.device)[ids].repeat_interleave(pipeline.num_frame_per_block, dim=1)
    noise = torch.randn(gt.shape, device=gt.device, dtype=gt.dtype, generator=rng)
    noisy = pipeline.scheduler.add_noise(gt.flatten(0, 1), noise.flatten(0, 1), times.flatten()).unflatten(0, gt.shape[:2])
    # Clear shape/mode-specific cached mask before each TF forward.
    transformer = pipeline.generator.model
    if hasattr(transformer, "get_base_model"):
        transformer = transformer.get_base_model()
    transformer.block_mask = None
    flow, prediction = pipeline.generator(noisy, conditioning, times, clean_x=clean_context)
    mask = torch.zeros_like(gt, dtype=torch.bool)
    mask[:, index + 1:] = True
    loss = get_denoising_loss("flow")()(x=gt.float(), x_pred=prediction.float(), noise=noise.float(),
                                      noise_pred=None, alphas_cumprod=None, timestep=times,
                                      flow_pred=flow.float(), gradient_mask=mask)
    delta = corrected.float() - history[:, -1:].float()
    return loss + regularization * delta.square().mean()
