# FM-Only A: Training Start Audit

Recorded on October 22, 2026 (Asia/Shanghai). This is a **start/preflight audit**,
not a completed epoch or benchmark result. The durable service is
`wan-fm-control-20261022.service` and the independent code is in
`experiments/wan_fm_control/`.

## Protocol

- Original Wan2.1-T2V-1.3B, native unchanged `FlowMatchSFTLoss` only.
- Same C config and initial LoRA (rank/alpha 8, q/k/v/o), LR 1e-4, AdamW,
  global batch 4, seed 42, deterministic attention and gradient checkpointing.
- 3995 training clips, one epoch, 999 optimizer updates; held-out 498 validation
  and sealed 500 dataset test clips never enter the optimizer.
- No Teacher, projector, Grounding or SAID loss, and zero disk feature cache.
- Independent four-GPU native-geometry preflight: 49 x 480 x 720, two updates,
  all 30 FM gradient blocks, all-rank synchronized adapter checks.

## Resume Check And Correction

The initial byte-hash equality check stopped before full training. Both smoke
checkpoints, raw logs and original lock/status are preserved. Exact sample
losses/noise/timesteps matched; only 3 of 480 adapter tensors differed, maximum
1.0913936421275139e-11. Optimizer maximum difference was 1.4155343563970746e-15.
`preflight_migration.json` records the explicit lock migration. The native
trainer, objective, preprocessing and C configuration were unchanged.

The corrected gate requires exact samples/crops/noise/timesteps/losses, LoRA
rtol 1e-7/atol 1e-10, optimizer rtol 1e-7/atol 1e-12 with exact step counters.
The report says `resume_numerically_equivalent=true`, **`resume_exact=false`**.
It does not claim byte-identical DDP resume.

Full training starts afresh, not from either smoke adapter. The first full A
checkpoint matches C's initial LoRA hash and all four first-update native FM
losses exactly. Later A/C losses are expected to diverge as their objectives
differ. `initial_control_check.json` preserves the per-rank values.

## Queue And Evaluation

The existing Base/C generation queue was temporarily stopped, retaining its
64 verified C VBench videos. A failure of the initial gate automatically
restored that queue; the corrected launch paused it again. No unrelated user
process was stopped. The service's failure/termination handler restores the
previously active unfinished queue.

Automatic order: A one-epoch training -> 498 held-out FM diagnostics -> same
100 VideoPhy-2 prompts/seeds and official SA/PC/Joint@4 -> interim Base/A/C table
and GitHub audit -> restore/finish Base/C VBench -> same 100 VBench annotated
prompt groups x five seeds for A -> all 16 official dimensions/aggregation ->
complete Base/A/C comparison and GitHub audit. All generation parameters and
scorer weight locks are unchanged. Registered adapter hashes must match the
scored adapters. No new A benchmark score is available at this snapshot.

## Verification And Limits

132 tests passed: 18 shared (including real downloaded-video integration),
11 download, 78 existing pilot, 25 independent FM control tests. Logs and
source/artifact SHA256 manifests are included. The active training/parent queue
source/input locks were checked unchanged when this audit was captured.

These are exploratory, single-training-seed, 100-prompt subset experiments,
not full official benchmark results or full-parameter fine-tuning. Human review
remains pending. The meaningful alignment comparison is C minus FM-only A;
Base versus C alone does not isolate the effect of SAID. Official color
conditional exclusions remain explicit and are never imputed as zero scores.

Only small audit files are uploaded. Checkpoints, videos, model weights and
datasets remain local.
