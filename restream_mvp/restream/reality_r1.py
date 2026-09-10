"""R1-A inputs for matched fresh adapters; no optimizer or checkpoint loading."""
import torch
from torch.nn import functional as F
from .reality_router import FrozenStateRouter
from .reality_temporal import assert_visible_frame_indices, pixel_count_for_latent_prefix


@torch.no_grad()
def encode_prefix(encoder, pixels, prefix_latents, count=3):
    """B,3,T,H,W -> B,(frames*P),D, using only the causally visible prefix."""
    visible = pixel_count_for_latent_prefix(prefix_latents)
    if pixels.ndim != 5 or pixels.shape[1] != 3 or pixels.shape[2] < visible:
        raise ValueError('Insufficient visible prefix pixels')
    if type(count) is not int or not 1 <= count <= visible:
        raise ValueError('Invalid prefix query frame count')
    indices = ([visible - 1] if count == 1 else
               torch.linspace(0, visible - 1, count).round().long().tolist())
    assert_visible_frame_indices(indices, prefix_latents)
    images = ((pixels[:, :, indices].transpose(1, 2).float() + 1) / 2).clamp(0, 1)
    # Concatenation preserves the per-image tokens; only router pooling averages.
    return encoder.encode_visual(images).flatten(1, 2), indices


class R1CandidateBank:
    """Same-split async positives + hardest cross-source async donor.

    Hard donors depend only on correct reference content, never on the query.
    Source labels construct this diagnostic proxy; the router only sees tensors.
    """
    def __init__(self, dataset, reference_count=2):
        if type(reference_count) is not int or reference_count < 1:
            raise ValueError('Candidate reference_count must be positive')
        self.dataset, self.count = dataset, reference_count
        self.pools = [row['reference_sets']['async'][:reference_count] for row in dataset.rows]
        if any(len(pool) != reference_count for pool in self.pools):
            raise ValueError('Insufficient async references for an R1 candidate bank')
        self.means = torch.stack([dataset.reference_features(pool).float().mean((0, 1)) for pool in self.pools])

    def __getitem__(self, index):
        row = self.dataset.rows[index]
        similarity = F.cosine_similarity(self.means[index][None], self.means, dim=-1)
        for i, donor in enumerate(self.dataset.rows):
            if donor['source_id'] == row['source_id'] or donor['sha256'] == row['sha256'] or donor['split'] != row['split']:
                similarity[i] = -torch.inf
        if not torch.isfinite(similarity).any():
            raise ValueError('No same-split, cross-source hard donor')
        donor = int(similarity.argmax())
        refs = self.pools[index] + self.pools[donor]
        return self.dataset.reference_features(refs)[None], refs, donor


def select_r1_memory(branch, prefix, candidates, reference_count=2, global_async=None, temperature=.1):
    if candidates.ndim != 4 or candidates.shape[1] != 2 * reference_count:
        raise ValueError('Expected correct async K + hard wrong K candidate bank')
    if branch == 'routed':
        return FrozenStateRouter(temperature=temperature)(prefix, candidates)
    if branch == 'correct_only':
        features = candidates[:, :reference_count]
    elif branch == 'global_async':
        if global_async is None or global_async.shape != candidates.shape[-2:] or not torch.isfinite(global_async).all():
            raise ValueError('Global baseline requires a finite train-only async mean')
        features = global_async[None, None].expand(candidates.shape[0], reference_count, -1, -1)
    else:
        raise ValueError('Unknown R1 branch')
    mask = torch.ones(features.shape[:2], device=features.device, dtype=torch.bool)
    return features, mask, {'active': mask.any(-1)}
