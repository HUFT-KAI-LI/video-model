# Experiment 1: FM + SAID Alignment

Status: implemented and smoke-tested; research efficacy is not established.
This project implements only the first module in the experiment specification.
It does not implement Unsaid, complementary learning, or temporal-relation losses.

## Isolation and Reuse

All experiment code and tracked evidence live under `experiments/wan_said/` on
`codex/wan-said-exp1`. The parent baseline is
`450d6f2` and the original `main` worktree remains separate. Do not move this
training implementation into LongLive or Reality Memory, or stage their pending edits.

| Component | Source and ownership | Modification policy |
| --- | --- | --- |
| Wan DiT, original umT5, original VAE | Pinned DiffSynth-Studio loaders and native forward | Reuse; no vendor edits or second DiT |
| FM loss, noise, timestep sampling, target, weights | `diffsynth.diffusion.loss.FlowMatchSFTLoss`, `FlowMatchScheduler('Wan')` | Import unchanged; no local FM formula |
| Normal video generation | Native DiT forward, Wan scheduler, VAE decoder | No Teacher, gate, or auxiliary head at inference |
| VideoMAEv2 | Pinned OpenGVLab implementation and checksum-verified weights | Frozen; no vendor edits |
| LoRA | Existing PEFT implementation | q/k/v/o adapters, rank 8, all 30 blocks |
| Grounding, joint semantic gate, student projection | `alignment.py` | New, experiment-local modules |
| Data, caches, training and evaluation orchestration | Other modules in this directory | Experiment-local; share frozen weights read-only |

The exact upstream revisions and Teacher checksum are in `upstream_lock.json`.
`runtime.upstream()` checks both vendor revisions and rejects dirty vendor trees.
Original Wan weights are loaded strictly; only LoRA parameters are trainable in DiT.
Weights, downloaded videos, caches and run artifacts are excluded from Git.

## Features and Gate

Original umT5 supplies Wan's normal text conditioning without replacement by CLIP.
The auxiliary text input is padding-aware pooled umT5 features:

```text
t_j = normalize(P_T(masked_pool(umT5(prompt_j))))
v_i = P_V(pool(VideoMAEv2(video_i)))
m_ij = sigmoid(MLP([t_j, v_i, t_j * v_i, abs(t_j - v_i)]))
```

The projection space has 256 channels. Each projector is LayerNorm, Linear,
GELU, Linear. The gate is a channel-wise semantic selector, not pixel segmentation.
The implementation supports an optional hard straight-through gate; the current
training configurations use a soft gate.

Only frozen raw Teacher/text features and VAE latents are cached. Trainable
projector outputs are recomputed. Teacher and Wan share the same 17-frame,
8-fps video window and center crop; Teacher takes 16 recorded indices from that
window. Fixed video grids contain no padded video tokens. `masked_pool` supports
valid-token masks when needed; variable-length padded grids are not currently used.

## Stage 1: Grounding

VideoMAEv2 and umT5 stay frozen. Train `P_T`, `P_V`, and the gate with train-only
cached features and a real batch of 64 examples on the expanded data:

```text
s_ij = cosine(normalize(m_ij * v_i), t_j) / 0.07
L_contrast = (CE(s, diagonal_labels) + CE(s.T, diagonal_labels)) / 2
L_sparse = mean(soft_mask) + 10 * mean((channel_mean(soft_mask) - 0.5)^2)
L_ground = L_contrast + 0.01 * L_sparse
```

All positive and negative candidates use the same pair-conditioned gate.
Same-source or normalized-duplicate-caption pairs are excluded as negatives.
Each row and column must retain a valid negative. Gradient accumulation is not
treated as a larger contrastive batch. Unrecognized semantic paraphrases remain
a documented false-negative risk, not a solved problem.

Full, random and shuffled-text controls plus heldout retrieval, text/video
sensitivity and keep ratios diagnose failure. Keep-budget regularization is not
a guarantee against shortcuts. Chance retrieval or nearly text-invariant gates
do not establish semantic selection and must be resolved before an efficacy pilot.

## Stage 2: Native FM plus Distillation

After Grounding pretraining, freeze the entire target branch. For a matched pair:

```text
target_i = stop_gradient(normalize(m_ii * v_i))
student_i = P_D(pool(DiT.blocks[14](noisy_latent, timestep, normal_text_condition)))
L_distill = mean(1 - cosine(student_i, target_i))
L_train = native_FM_loss + 0.05 * L_distill
```

This is explicitly a staged scheme, not end-to-end optimization of Grounding
and DiT. `L_ground` is not part of the Stage 2 optimizer. A configurable
`distill_max_sigma` may restrict the auxiliary loss to lower-noise examples;
the default 1.0 includes every sampled noise level without changing FM sampling.
Noise-bucket diagnostics are recorded. All-zero hard targets give differentiable
zero auxiliary supervision instead of NaNs.

The Teacher only supplies a loss target. It never enters the native DiT forward
as conditioning. H15 is captured without detach, the hook is removed before
backward, and pooling is outside checkpoint recomputation. Native velocity
prediction still runs through all 30 blocks.

| Objective | Parameters receiving gradients |
| --- | --- |
| Grounding only | Text/video projectors and joint gate; no frozen encoder or DiT |
| FM only | LoRA in blocks 1-30; no student projection or Grounding |
| Distillation only | Student projection and LoRA in blocks 1-15; no blocks 16-30 or target branch |

`tests/test_native_contract.py` compares native FM with and without an H15 hook
using identical RNG, including exact loss, timestep, noisy inputs, velocity and
parameter-gradient equality. This is a CPU regression on a tiny student, not a
replacement for the real Wan smoke/gradient checks in `reports/`.

## Matched Controls and Inference

| Arm | Grounding preparation | DiT objective |
| --- | --- | --- |
| A | None | Native FM only |
| B | Independent unmasked projectors; same batch/steps as C | FM + unmasked Teacher distillation |
| C | Joint-gated Grounding | FM + SAID-selected Teacher distillation |

Initial Wan and LoRA weights, video order, seeds, LoRA settings, DiT step budgets,
prompts and inference settings match. Grounding adds cost to B/C relative to A;
it is disclosed separately rather than claiming equal end-to-end compute.
Original umT5, DiT with the chosen LoRA, native scheduler and VAE are the only
inference modules. Clip-level alignment does not by itself prove object/attribute
binding, action binding, or improved motion.

## Data and Evidence

The downloaded OpenVid selection contains 5,000 clips from distinct parsed source
IDs. Seven near duplicates were removed before the frozen split: **3,995 train,
498 validation, 500 test**. Manifests and their SHA256 lock are tracked in
`manifests_openvid/`. Raw selected videos total 13.27 GB; this is separate from
feature-cache storage. The planning code caps raw data and checks a conservative
100-GB total-data budget. No full source ZIP archives are stored.

Optimizer batches, Grounding negatives and Teacher targets use train only.
Validation is for diagnostics and selection. Test is sealed; test features are
not computed during preparation. Source grouping, hashes, captions and pHash
checks reduce leakage but cannot establish independence from Wan pretraining or
from all related episodes/channels.

The old 32/6/6 split is smoke-only. It supports the existing A/B/C matched
5-step real-Wan checks and 12 native-path generated videos, not an efficacy
claim. Expanded-data Grounding validation, the matched 300-step pilot and
quantitative native-generation evaluation remain pending. Report failures and
negative results; do not add Unsaid before validating this first module.
