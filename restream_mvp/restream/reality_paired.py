"""Same-target R0 pairing and a mild, reference-independent prefix degradation.

Labels select pairs and define a loss only. R0 still queries memory from prompt
context; neither source IDs nor video-state features enter the memory model.
"""
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from .corruption import corrupt_history
from .objective import future_loss
from .reality_runtime import PreserveHistory


def load_global_constant(config, cache, device):
    path = Path(config["reality_memory"]["references"]["global_constant_features"])
    if not path.is_absolute():
        from .runtime import ROOT
        path = ROOT / path
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("cache_identity") != cache.identity or payload.get("tokens") != cache.tokens or payload.get("channels") != cache.channels:
        raise ValueError("Global constant feature cache identity/shape mismatch")
    if payload.get("source_split") != "train" or not torch.isfinite(payload["features"]).all():
        raise ValueError("Global constant must be finite and computed from train references")
    return payload["features"].to(device)


def validate_paired_config(config):
    memory = config["reality_memory"]
    cfg = memory["objective"]["paired"]
    if cfg["correct_kind"] not in ("async", "aligned") or type(cfg["reference_count"]) is not int or cfg["reference_count"] < 1:
        raise ValueError("Paired training requires async/aligned correct references and positive K")
    if not math.isfinite(cfg["temperature"]) or cfg["temperature"] <= 0 or not math.isfinite(cfg["contrast_weight"]) or cfg["contrast_weight"] < 0:
        raise ValueError("Invalid paired contrast weight/temperature")
    if memory["regularization"]["wrong_gate_weight"] != 0 or memory["references"]["per_reference_dropout"] != 0:
        raise ValueError("Controlled pairs require zero one-sided wrong-gate penalty and zero reference dropout")
    degradation = cfg["degradation"]
    if degradation["mode"] not in ("clean", "mild"):
        raise ValueError("Prefix mode must be clean or mild")
    if type(degradation["protected_prefix"]) is not int or not 3 <= degradation["protected_prefix"] < memory["objective"]["prefix_latents"]:
        raise ValueError("Mild degradation must protect the first three latents and leave a nonempty suffix")
    if not math.isfinite(degradation["sigma"]) or not 0 < degradation["sigma"] <= .12:
        raise ValueError("Mild degradation requires 0 < sigma <= 0.12")
    seeds = cfg["probe_noise_seeds"]
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Paired probes require distinct nonnegative integer noise seeds")


def paired_references(row, cfg):
    count, kind = cfg["reference_count"], cfg["correct_kind"]
    correct, wrong = row["reference_sets"][kind][:count], row["reference_sets"]["wrong"][:count]
    if len(correct) != count or len(wrong) != count:
        raise ValueError("Both sides of a pair must have the configured reference count")
    for refs, same in ((correct, True), (wrong, False)):
        for ref in refs:
            if ref["split"] != row["split"] or (ref["source_id"] == row["source_id"]) != same:
                raise ValueError("Paired reference source/split leakage")
            if same and kind == "async" and ref["time"] >= row["target_start"]:
                raise ValueError("Async correct references must precede the target")
    return correct, wrong


class PairedRealityDataset(Dataset):
    def __init__(self, dataset, cfg, indices):
        self.dataset, self.cfg, self.rows = dataset, cfg, dataset.rows
        for index in indices:
            for refs in paired_references(self.rows[index], cfg):
                for ref in refs:
                    dataset.cache.read(ref)  # Fail before expensive backbone loading.

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        correct, wrong = paired_references(self.rows[index], self.cfg)
        return {**sample, "correct_features": self.dataset.reference_features(correct),
                "wrong_features": self.dataset.reference_features(wrong)}


def paired_history(gt, index, cfg, device, seed, mode=None):
    prefix = gt[:, :index + 1]
    degradation = cfg["degradation"]
    mode = mode or degradation["mode"]
    if mode == "clean":
        return prefix.clone()
    if mode != "mild":
        raise ValueError("Unknown prefix mode")
    return corrupt_history(prefix, torch.Generator(device=device).manual_seed(seed), probability=1,
                           sigma_min=degradation["sigma"], sigma_max=degradation["sigma"],
                           protected_prefix=degradation["protected_prefix"], spatial_shift_probability=0)


def contrast_loss(correct_score, wrong_score, temperature):
    # Two-class contrastive objective; no constraint that the final gate must open.
    return F.softplus((wrong_score.float() - correct_score.float()) / temperature).mean()


def video_context_gradient(loss_fn, context):
    """Exact first-order VJP through one frozen video pass; free it before the next.

    Holding two LongLive backward graphs exceeds an 80 GB GPU. Treat context as
    a temporary leaf, then chain its derivative through the small paired memory
    graph. The numerical objective and parameter gradients are unchanged; this
    is not a straight-through estimator or a higher-order derivative API.
    """
    leaf = context.detach().requires_grad_(True)
    loss = loss_fn(leaf)
    gradient, = torch.autograd.grad(loss, leaf)
    return loss.detach(), gradient.detach()


def paired_loss(pipeline, model, gt, conditioning, index, batch, rng, config):
    cfg = config["reality_memory"]["objective"]["paired"]
    device = gt.device
    # Independent seeds; both references get exactly the same history/times/noise.
    seeds = torch.randint(0, 2**31, (2,), device=device, generator=rng).tolist()
    history = paired_history(gt, index, cfg, device, seeds[0])
    features = torch.cat((batch["correct_features"], batch["wrong_features"]), 0).to(device)
    context = conditioning["prompt_embeds"]
    fused, stats = model(context.expand(2, -1, -1), features,
                         torch.ones(features.shape[:2], dtype=torch.bool, device=device))
    values, gradients = [], []
    for side in (0, 1):
        def objective(embedding):
            return future_loss(pipeline, PreserveHistory(), gt, history, history[:, -1:],
                               {**conditioning, "prompt_embeds": embedding}, index,
                               torch.Generator(device=device).manual_seed(seeds[1]), 0)
        value, gradient = video_context_gradient(objective, fused[side:side + 1])
        values.append(value)
        gradients.append(gradient)
    # .5*(L_correct + L_wrong) trains both sides toward the same GT future.
    video = torch.stack(values).mean()
    chain = (fused.float() * torch.cat(gradients).float()).sum() * .5
    video_with_grad = video + (chain - chain.detach())
    contrast = contrast_loss(stats["relevance_score"][0], stats["relevance_score"][1], cfg["temperature"])
    regularization = config["reality_memory"]["regularization"]["delta_weight"] * stats["delta_square"]
    loss = video_with_grad + cfg["contrast_weight"] * contrast + regularization
    parameters = tuple(model.parameters())
    video_grads = torch.autograd.grad(video_with_grad, parameters, retain_graph=True, allow_unused=True)
    contrast_grads = torch.autograd.grad(contrast, parameters, retain_graph=True, allow_unused=True)
    def grad_norm(grads):
        values = [g.float().square().sum() for g in grads if g is not None]
        return torch.stack(values).sum().sqrt() if values else loss.new_zeros(())
    weighted_contrast = cfg["contrast_weight"] * grad_norm(contrast_grads)
    return loss, {**stats, "video_loss": video, "wrong_loss": video.new_zeros(()),
                  "correct_video_loss": values[0], "wrong_source_video_loss": values[1],
                  "contrast_loss": contrast.detach(), "history_seed": seeds[0], "noise_seed": seeds[1],
                  "video_gradient_norm": grad_norm(video_grads).detach(),
                  "contrast_gradient_norm_raw": grad_norm(contrast_grads).detach(),
                  "contrast_gradient_norm_weighted": weighted_contrast.detach(),
                  "prefix_suffix_mse": (history[:, 3:].float() - gt[:, 3:index + 1].float()).square().mean()}
