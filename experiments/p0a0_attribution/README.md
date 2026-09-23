# P0-A0 — Base Model Attribution Test

Use two free GPUs for the paired 20 prompts × 4 seeds × 2 models × 5 seconds experiment.
The 60-second Mini-Pilot is deliberately not started by this experiment.

Wan uses the unmodified official inference source at commit
`9737cba9c1c3c4d04b33fcad41c111989865d315` (vendored under `vendor/wan21`, Apache 2.0).
Its [official instructions](https://github.com/Wan-Video/Wan2.1/tree/9737cba9c1c3c4d04b33fcad41c111989865d315)
recommend 480P, CFG 6 and shift 8–12; this experiment fixes shift 8 and 50 UniPC steps.
LongLive uses the official base + LoRA, four denoising steps and its causal cache.
No prompt expansion, selection by quality, automatic rerolls, or long-video pilot.

## Frozen design

- Exact 20 existing controlled prompts, seeds 0–3, 80 paired conditions.
- 832×480, 16 FPS. Both generate 21 latents / 81 native frames; save frames 0–79 (5 s).
- Primary: **human Initial Attribute Accuracy in frames 0–15, the first second**.
  `CORRECT`: requested subject/item is present and has the requested color throughout
  visible parts of that interval. `INCORRECT`: visible wrong color/item or obvious omission.
  `UNJUDGEABLE`: hidden, too small, ambiguous, or not yet visible. Blank means pending.
  Do not convert pending or unjudgeable to an incorrect label silently.
- Report judgeable accuracy, coverage, and lower/upper bounds over all 80 samples.
  Object absence, attribute mismatch and later change have separate review fields.
  Whole 5 s clips are provided for context; do not change an initial label because of later frames.
- Automatic auxiliary proxy: fixed CLIP ViT-B/32, whole frames 0, 4, 8, 12;
  choose highest mean cosine among eleven color substitutions in the target phrase.
  Subject color is held fixed (e.g. **white dog** remains white in all harness options).
  Yellow/orange are included. This proxy cannot establish object presence or localization.
- Later proxy: frames 16, 32, 48, 64, 79; kept separate from primary initial accuracy.
- Same numeric seed is a pairing key, **not identical input noise**: Wan and LongLive
  have different RNG layouts, dtypes and stochastic trajectories.
- Any gap describes the two released inference pipelines. Distillation, CFG, negative
  prompting, sampler, precision and causal attention differ; do not isolate causalization
  or conclude equivalence from a nonsignificant difference.
- The 20 prompts reuse eight semantic/motion templates. Exploratory uncertainty uses
  a paired cluster bootstrap over those eight groups, not independent video resampling.
- Initial mismatches remain in the dataset. Recoverability requires the separate
  initially-correct → later-failure criterion.

## Run

For the authorized four-GPU schedule, run `bash experiments/p0a0_attribution/run_four_gpus.sh`.
The script verifies model hashes, runs all four workers, and finalizes only after completion.

Use the repository virtualenv `restream_mvp/.venv/bin/python` and its installed CUDA stack.
`generate.py` freezes its source hashes and design, archives exact source files, claims jobs
atomically, checks completed checksums, and preserves failed attempts without automatic retry.
Only a fresh run directory may adopt changed generation code. The preliminary `profile_v1`
run used the LongLive-vendored Wan implementation and is excluded from formal results.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  restream_mvp/.venv/bin/python experiments/p0a0_attribution/generate.py \
  --run experiments/p0a0_attribution/runs/a0_20260923 --model wan
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  restream_mvp/.venv/bin/python experiments/p0a0_attribution/generate.py \
  --run experiments/p0a0_attribution/runs/a0_20260923 --model longlive
```

Run Wan on GPU 0, 2 and 3; LongLive on GPU 1. After LongLive finishes, GPU 1 also
claims remaining Wan jobs from the same run directory. Atomic claims prevent duplicate
samples. All four GPUs are authorized for this experiment.
Keep the model and text encoder resident where possible; offload Wan's text encoder only
for the FP32 VAE decode peak. LongLive's measured reserved memory is about 20 GiB.

```bash
restream_mvp/.venv/bin/python experiments/p0a0_attribution/evaluate.py --run RUN
restream_mvp/.venv/bin/python experiments/p0a0_attribution/review.py --run RUN
restream_mvp/.venv/bin/python experiments/p0a0_attribution/summarize.py --run RUN
# After actual human review:
restream_mvp/.venv/bin/python experiments/p0a0_attribution/summarize.py --run RUN --labels labels.csv
```

`review/index.html` uses randomly ordered anonymous filenames; its separate unblinding map
must not be given to reviewers. Exported labels do not overwrite raw videos or automatic scores.
Full videos, source archives, latents and weights remain local; compact evidence and results
are published on the review branch. Do not publish a human accuracy before reviewing labels.
