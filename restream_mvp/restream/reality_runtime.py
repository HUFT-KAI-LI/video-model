"""R0 integration reuses LongLive loading, teacher forcing, AR replay and target decoding."""
import copy
import hashlib
from pathlib import Path
import torch
from torch import nn
from .objective import future_loss
from .reality_cache import FeatureCache
from .reality_dataset import RealityDataset
from .reality_encoder import encoder_identity
from .reality_memory import RealityMemory, reference_dropout, memory_regularization
from .runtime import ROOT, read_config


def read_reality_config(path):
    config = read_config(path)
    memory = config["reality_memory"]
    if memory["stage"] != "r0" or memory["guidance"]["mode"] != "context":
        raise ValueError("Only R0 context guidance is implemented; R1 awaits R0 results")
    if not config["train"]["backbone_frozen"] or not memory["encoder"]["frozen"] or not memory["guidance"]["zero_init"]:
        raise ValueError("R0 requires frozen backbones and zero-initialized residual output")
    if memory["encoder"]["type"] != "dinov2":
        raise ValueError("R0 uses one frozen DINOv2 encoder")
    if config["data"]["frames"] < 21 or not 0 <= memory["references"]["per_reference_dropout"] <= 1:
        raise ValueError("Invalid R0 temporal window/dropout")
    prefix = memory["objective"]["prefix_latents"]
    if prefix < 3 or prefix % 3 or prefix + 3 > (config["data"]["frames"] - 1) // 4 + 1 or memory["objective"]["future_blocks"] != 1:
        raise ValueError("R0 supervises exactly one future AR block after a block-aligned prefix")
    if "paired" in memory["objective"]:
        from .reality_paired import validate_paired_config
        validate_paired_config(config)
    return config


def make_memory(config):
    memory = config["reality_memory"]
    return RealityMemory(memory["encoder"]["feature_dim"], memory["projector"]["memory_dim"],
                         memory["guidance"]["context_dim"], memory["guidance"]["heads"],
                         memory["guidance"]["gate_init_logit"])


def make_cache(config):
    memory = config["reality_memory"]
    identity = encoder_identity(ROOT / memory["encoder"]["path"], memory["encoder"]["image_size"], memory["projector"]["num_memory_tokens"])
    return FeatureCache(ROOT / memory["feature_cache"], identity, memory["projector"]["num_memory_tokens"], memory["encoder"]["feature_dim"])


def make_dataset(config, split, cache=None):
    data = config["data"]
    return RealityDataset(ROOT / data[f"{split}_manifest"], cache or make_cache(config),
                          data["frames"], data["height"], data["width"], data["fps"])


@torch.no_grad()
def prepare_reality(pipeline, batch, device, config):
    if batch["pixels"].shape[0] != 1:
        raise ValueError("R0 currently requires per-GPU batch=1")
    gt = pipeline.vae.encode_to_latent(batch["pixels"].to(device, dtype=torch.bfloat16)).to(torch.bfloat16)
    prefix = config["reality_memory"]["objective"]["prefix_latents"]
    if prefix % pipeline.num_frame_per_block or prefix + pipeline.num_frame_per_block > gt.shape[1]:
        raise ValueError("Prefix/future must align with actual model AR block size")
    conditioning = pipeline.text_encoder(batch["caption"])
    return gt, conditioning, prefix - 1


class PreserveHistory(nn.Module):
    def forward(self, predicted, unused_reference):
        return predicted


def reality_loss(pipeline, model, gt, conditioning, index, features, mask, wrong, rng, config, training=True):
    memory = config["reality_memory"]
    mask = reference_dropout(mask, rng, memory["references"]["per_reference_dropout"], training)
    fused, stats = model(conditioning["prompt_embeds"], features, mask)
    cond = {**conditioning, "prompt_embeds": fused}
    # Reuse the verified first-future-block flow loss without modifying any latent.
    video = future_loss(pipeline, PreserveHistory(), gt, gt[:, :index + 1], gt[:, index:index + 1], cond, index, rng, 0)
    regularization, wrong_loss = memory_regularization(stats, wrong, memory["regularization"]["delta_weight"],
                                                       memory["regularization"]["wrong_gate_weight"])
    return video + regularization, {**stats, "video_loss": video.detach(), "wrong_loss": wrong_loss.detach()}


def resume_signature(config, cache):
    # Scheduling extensions are allowed; optimizer/data/model/dropout semantics must match.
    semantic = copy.deepcopy(config)
    for name in ("max_steps", "max_updates", "save_every", "eval_every"):
        semantic["train"].pop(name, None)
    return {"config": semantic, "encoder": cache.identity,
            "manifests": {split: hashlib.sha256((ROOT / config["data"][f"{split}_manifest"]).read_bytes()).hexdigest()
                          for split in ("train", "val")}}
