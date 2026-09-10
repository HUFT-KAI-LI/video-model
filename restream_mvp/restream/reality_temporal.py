"""Wan causal temporal mapping shared by manifests, datasets and diagnostics.

Wan groups pixel frames as ``[0]``, ``[1..4]``, ``[5..8]``, ...: the first frame
is its own latent and each following latent covers four more frames. Therefore
``L`` prefix latents expose exactly ``4*(L-1)+1`` pixel frames with indices
``0..4*(L-1)``, and the last visible pixel-frame index is ``4*(L-1)``.

Getting this wrong by one latent group silently turns a state query into a
look-ahead (a retrieval query that sees the future), so every caller - Dataset
prefix validation, the manifest builder and the retrieval diagnostics - uses
these functions instead of open-coding ``4 * prefix_latents``.
"""


def pixel_count_for_latent_prefix(latent_count):
    """Number of pixel frames visible to a prefix of ``latent_count`` latents."""
    if type(latent_count) is not int or latent_count < 1:
        raise ValueError("latent_count must be a positive integer")
    return 4 * (latent_count - 1) + 1


def latent_prefix_boundary_index(latent_count):
    """Index of the last pixel frame visible to the prefix (4*(L-1))."""
    return pixel_count_for_latent_prefix(latent_count) - 1


def prefix_visible_seconds(prefix_latents, fps):
    """Time from the window start to the last visible pixel frame."""
    if not fps or fps <= 0:
        raise ValueError("fps must be positive")
    return latent_prefix_boundary_index(prefix_latents) / float(fps)


def assert_visible_frame_indices(frame_indices, prefix_latents, label="frame indices"):
    """Reject any index beyond the causal prefix boundary."""
    limit = latent_prefix_boundary_index(prefix_latents)
    if any(index < 0 or index > limit for index in frame_indices):
        raise ValueError(f"{label} {list(frame_indices)} exceed the visible prefix boundary {limit} "
                         f"for {prefix_latents} latents")
    return list(frame_indices)
