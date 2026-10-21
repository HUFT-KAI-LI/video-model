# Review Record: 2026-10-21

Review branch: `codex/wan-said-pilot-v2`, based on `37ab1bb`.
All new tracked changes are confined to this `pilot_v2` directory. Shared
alignment/FM/runtime files, old reports, and the user's dirty LongLive/Reality
worktree are not edited or included in this commit.

## What Is Actually Complete

- The existing video-level Experiment 1 computation is reused with ViT-Base,
  rectangular Teacher input, full decoded-PTS sampling, and an online-only data
  provider. No Unsaid/TRD/token loss is introduced.
- All 4,993 retained pilot source files were SHA256-verified when freezing the new
  manifests. Original train 3,995 / val 498 / test 500 membership is unchanged.
- Eight train/val videos were audited at native source sampling for each of
  17/49/81 Wan frames, producing 16/48/80 Teacher frames. Every shape is valid,
  outputs are finite/frozen, and repeated forwards match exactly. The default
  result is 3,600 x 768 raw tokens. Checkpoint loading is strict after the four
  explicitly documented classifier/fc_norm removals.
- Final deterministic Full/Said Grounding checks ran 3 optimizer steps each,
  using an eight-video training pool and four heldout validation videos.
- Final A/B/C checks ran **2 real optimizer steps per arm at 49x480x720**. Only
  two unique training videos per arm were presented; this is not a full pass over
  the 3,995 training videos. Initialization, sequence, latent SHA256, noise seeds,
  timesteps and preprocessing match across arms.
- Separate backward audits verify all 30 blocks for FM and blocks 1-15 for H15
  distillation. Frozen encoders/Grounding receive no gradients. A's projector
  receives no FM gradients.
- A's 2-step checkpoint resumed to step 3 matches an uninterrupted 3-step run
  **byte-for-byte** in LoRA state: hash
  `5b48bab5583cf90e4c2e106cca9acabc221f7280f99f362a1f6c5dec27716eaf`.
  C also resumed to step 3 and Said Grounding to step 4 with optimizer states.
  Original checkpoints were not overwritten.
- Native UniPC inference exported 12 genuine videos: base/A/B/C, one original
  prompt from each of the three benchmark suites. Geometry is 17x256x384,
  4 steps, 16 FPS, CFG 6, shift 8. Outputs decode, are nonblank, and have changing
  pixels. They are **not** 81-frame/480P formal evaluations and prove no efficacy.
  Existing C videos were reopened and hash/geometry-validated by a second run.
- New feature cache bytes: zero. Encoder weights and optimizer/adapter checkpoints
  still occupy storage; "no cache" does not mean no training storage whatsoever.

Machine-readable evidence and final test transcripts are in `reports/20261021/`.

## Observed And Corrected

The first resume test failed exact comparison. Inspection found that the installed
FlashAttention 2.8.3 default backward kernel is nondeterministic. The new scoped
execution adapter now selects its documented `deterministic=True` option and
enables deterministic PyTorch execution. The original forward, FM objective, and
shared source files are unchanged. The final resume equality above was measured
after this correction. Pre-correction runs are retained locally in
`runs/pilot_v2/smoke_20261021` and `runs/pilot_v2/resume_20261021`; do not mix them
with the final deterministic checkpoints.

An early new-test run also exposed two mistakes in the tests themselves (passing
unpooled tokens to video-level cosine loss and passing a path rather than split
records to the shared split audit). Those test calls were corrected; the shared
algorithm was not changed to accommodate them. Final transcripts are authoritative.

## Dataset And Protocol Caveats

- Pilot source selection was capped at 4 MB/video. It is not the official 64K
  distribution. Both 64K acquisition services were confirmed inactive/dead; this
  change does not restart them or delete partially downloaded videos.
- Duration statistics from frozen pilot metadata: P50 4.04s, P90 6.4s, P95
  7.618s, P99 10.501s, max 18s. Full-range 49-frame sampling can miss fast events;
  covering source endpoints is not equivalent to semantic coverage.
- The shared center crop often retains 84.375% of a 16:9 source area for 3:2
  training. It can remove relevant content. Named human review is still pending;
  the code does not invent approvals. A rejection blocks formal training until a
  corrected/filtered dataset has been frozen under a new identity.
- All 4,993 captions were token-length audited with the actual local native
  umT5 tokenizer: max 285 tokens, zero exceed Wan's 512-token limit.
- T2V-CompBench locks 1,400 original prompts plus seven metadata files. Two
  upstream spelling/grammar differences are recorded, not silently corrected.
- VBench's pinned standard file has 946 metadata rows but 944 distinct generation
  prompts. Five outputs per unique prompt gives 4,720 jobs without dropping any
  official metadata dimensions.
- Pinned VideoPhy2 files have 590/591 rows, not the requested 344. No arbitrary
  344-row subset is substituted; the discrepancy needs explicit resolution.

## Still Outstanding

Grounding's three-step heldout check is not validated for formal training: Said
R@1 is 0.25 video-to-text / 0.50 text-to-video on four samples, versus chance
0.25. Neither this tiny diagnostic nor the loss reduction proves semantic
selectivity. Formal B/C training rejects this checkpoint without `--smoke`.

Full Grounding pretraining, human caption/event/crop approval, formal A/B/C
training, official evaluator environments/weight locks, official scoring and
aggregation integration, and complete benchmark results remain unexecuted.
The scorer file is a pinned recipe/preflight interface, not a claim that all
three official scoring stacks are installed or successfully tested. No paper
score, improvement claim, or cross-benchmark proxy metric is produced.

## Executed Commands

All commands below ran in the isolated experiment worktree, using the existing
`experiments/wan_said/.venv/bin/python`. Prefix modules with that interpreter.
Run directories are ignored by Git; only small evidence artifacts are copied.

```bash
-m experiments.wan_said.pilot_v2.manifests
-m experiments.wan_said.pilot_v2.audit --output experiments/wan_said/runs/pilot_v2/audit_final_20261021 --count 8 --teacher-check
-m experiments.wan_said.pilot_v2.smoke --output experiments/wan_said/runs/pilot_v2/smoke_deterministic_20261021 --ground-steps 3 --train-steps 2
-m experiments.wan_said.pilot_v2.verify_runs --smoke-root experiments/wan_said/runs/pilot_v2/smoke_deterministic_20261021 --output experiments/wan_said/runs/pilot_v2/resume_deterministic_20261021
-m experiments.wan_said.pilot_v2.benchmarks --output experiments/wan_said/runs/pilot_v2/benchmark_plan_20261021
-m experiments.wan_said.pilot_v2.smoke_inference --smoke-root experiments/wan_said/runs/pilot_v2/smoke_deterministic_20261021 --output experiments/wan_said/runs/pilot_v2/inference_20261021
-m experiments.wan_said.pilot_v2.inference --jobs experiments/wan_said/runs/pilot_v2/inference_20261021/jobs.jsonl --output experiments/wan_said/runs/pilot_v2/inference_20261021/C --group C --checkpoint experiments/wan_said/runs/pilot_v2/smoke_deterministic_20261021/C/checkpoint.pt --smoke
```

`collect_reports.py` reruns all three test suites with the downloaded-video
integration test explicitly enabled, records their exit codes, and hashes review
artifacts and source files. Download tests use tiny synthetic fixtures and mock
transport; their fixture completion messages must not be read as 64K completion.

The raw training MP4s, Teacher/Wan weights, optimizer checkpoints and generated
evaluation MP4s are not uploaded. Test logs, per-step losses, gradient audits,
sampling timestamps, source/weight hashes, pending review sheet and eight preview
sheets are uploaded for code/operation review.
