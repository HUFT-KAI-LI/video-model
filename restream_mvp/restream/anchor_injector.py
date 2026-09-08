import math
import torch


def timestamp_to_latent_index(seconds, window_sec, pixel_frames, block_size=3):
    """Snap observation to the END of a causal VAE group and AR block.

    Wan encodes frame 0, then groups [1..4], [5..8], ... . Return the
    latent index and actual arrival time. This deliberately delays an anchor;
    it never labels an observation as having arrived before it was sampled.
    """
    if window_sec <= 0 or pixel_frames < 5 or (pixel_frames - 1) % 4 or block_size < 1:
        raise ValueError("Invalid Wan temporal shape/window")
    if not 0 <= seconds <= window_sec:
        raise ValueError("Anchor outside window")
    frame = math.ceil(seconds / window_sec * (pixel_frames - 1) - 1e-9)
    latent = math.ceil(frame / 4)
    latent = ((latent // block_size) + 1) * block_size - 1
    count = (pixel_frames - 1) // 4 + 1
    if latent >= count - 1:
        raise ValueError("Anchor must leave at least one future block")
    return latent, 4 * latent * window_sec / (pixel_frames - 1)


class HardAnchorInjector:
    def __init__(self, vae=None):
        self.vae = vae

    @torch.no_grad()
    def encode_anchor(self, image):
        """image: B,C,H,W, RGB [-1,1], same resize/crop as GT."""
        if self.vae is None or image.ndim != 4:
            raise ValueError("VAE and B,C,H,W image required")
        if not torch.isfinite(image).all() or image.min() < -1 or image.max() > 1:
            raise ValueError("Invalid normalized image")
        latent = self.vae.encode_to_latent(image.unsqueeze(2))
        if latent.ndim != 5 or latent.shape[1] != 1:
            raise ValueError("Expected one B,T,C,H,W image latent")
        return latent

    def inject(self, history, anchor, index, rebuild):
        """Rebuild MUST return fresh state; stale caches are never retained."""
        if history.ndim != 5 or not 0 <= index < history.shape[1]:
            raise ValueError("Invalid history/index")
        if anchor.shape != history[:, index:index + 1].shape:
            raise ValueError("Anchor shape mismatch")
        corrected = history.clone()
        corrected[:, index:index + 1] = anchor
        return corrected, rebuild(corrected)
