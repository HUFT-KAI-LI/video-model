import torch
from torch import nn
from torch.nn import functional as F


def masked_pool(tokens, valid=None):
    if tokens.ndim != 3:
        raise ValueError('Expected [batch, tokens, channels]')
    if valid is None:
        return tokens.float().mean(1)
    if valid.shape != tokens.shape[:2] or not valid.any(1).all():
        raise ValueError('Invalid token mask or empty sequence')
    weights = valid.to(dtype=torch.float32, device=tokens.device)
    return (tokens.float() * weights[..., None]).sum(1) / weights.sum(1, keepdim=True)


class Projector(nn.Sequential):
    def __init__(self, input_dim, dim=256):
        super().__init__(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim))


class JointGate(nn.Module):
    def __init__(self, dim=256, hard=False):
        super().__init__()
        self.hard = hard
        self.net = nn.Sequential(nn.Linear(4 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, text, video):
        text, video = torch.broadcast_tensors(text, video)
        soft = self.net(torch.cat((text, video, text * video, (text - video).abs()), -1)).sigmoid()
        if self.hard:
            return (soft > .5).to(soft.dtype) - soft.detach() + soft, soft
        return soft, soft


class Grounding(nn.Module):
    def __init__(self, text_dim=4096, video_dim=384, dim=256, hard=False):
        super().__init__()
        self.text = Projector(text_dim, dim)
        self.video = Projector(video_dim, dim)
        self.gate = JointGate(dim, hard)

    def encode(self, text, video):
        return F.normalize(self.text(text.float()), dim=-1), self.video(video.float())

    def scores(self, text, video, mode='said', temperature=.07, generator=None):
        t, v = self.encode(text, video)
        mask, soft = self.gate(t[None, :, :], v[:, None, :])
        if mode == 'full':
            mask = torch.ones_like(mask)
        elif mode == 'random':
            mask = (torch.rand(mask.shape, device=mask.device, generator=generator) > .5).float()
        elif mode == 'shuffled_text':
            mask, soft = self.gate(t.roll(1, 0)[None], v[:, None])
        elif mode != 'said':
            raise ValueError(mode)
        selected = F.normalize(mask * v[:, None], dim=-1, eps=1e-6)
        scores = (selected * t[None]).sum(-1) / temperature
        return scores, soft

    def loss(self, text, video, caption_groups, source_groups, mode='said', sparse_weight=.01):
        count = len(text)
        if count < 2 or len(caption_groups) != count or len(source_groups) != count:
            raise ValueError('Grounding requires a real multi-sample negative pool, not accumulation')
        scores, soft = self.scores(text, video, mode)
        excluded = torch.tensor([[i != j and (caption_groups[i] == caption_groups[j] or
                                              source_groups[i] == source_groups[j])
                                  for j in range(count)] for i in range(count)], device=scores.device)
        if ((~excluded).sum(1) < 2).any() or ((~excluded).sum(0) < 2).any():
            raise ValueError('Every contrastive row/column needs at least one valid negative')
        scores = scores.masked_fill(excluded, -torch.inf)
        labels = torch.arange(count, device=scores.device)
        contrastive = .5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels))
        keep = soft.mean(-1)
        # A budget anchor prevents uniform near-zero gates exploiting cosine scale invariance.
        sparse = soft.mean() + 10 * (keep - .5).square().mean()
        loss = contrastive + (sparse_weight * sparse if mode == 'said' else 0)
        return loss, dict(contrastive=contrastive.detach(), sparse=sparse.detach(),
                          keep=soft.diagonal(dim1=0, dim2=1).T.mean().detach())

    @torch.no_grad()
    def target(self, text, video, mode='said'):
        t, v = self.encode(text, video)
        mask, _ = self.gate(t, v)
        if mode == 'full':
            mask = torch.ones_like(v)
        elif mode != 'said':
            raise ValueError(mode)
        return F.normalize(mask * v, dim=-1, eps=1e-6).detach()


def distillation_loss(student, target):
    if student.shape != target.shape:
        raise ValueError(f'Distillation shape mismatch: {student.shape}, {target.shape}')
    target = target.detach().float()
    valid = target.norm(dim=-1) > 1e-6
    distances = 1 - F.cosine_similarity(student.float(), target, dim=-1, eps=1e-6)
    # Zero hard masks contain no supervision; return a differentiable finite zero.
    return (distances * valid).sum() / valid.sum().clamp_min(1)


class HiddenCapture:
    """One pooled tensor per forward; deactivate before checkpoint recomputation."""
    def __init__(self, block, valid=None):
        self.valid, self.hidden = valid, None
        self.handle = block.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if self.hidden is not None:
            raise RuntimeError('Block 15 called twice in a single capture scope')
        if not output.requires_grad:
            raise RuntimeError('H15 is detached; use non-reentrant gradient checkpointing')
        self.hidden = output

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.handle.remove()
        if exc[0] is None and self.hidden is not None:
            # Pool outside the checkpointed block so recomputation saves identical tensors.
            self.hidden = masked_pool(self.hidden, self.valid)
