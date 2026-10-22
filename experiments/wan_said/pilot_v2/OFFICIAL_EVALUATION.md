# C Official Benchmark Queue

## Active Request: 100-Video Pilot

Completed on 2026-10-22 at approximately 14:48 CST. All 100 paired scores are
valid: SA 3.16/5, PC 3.78/5, SA@4 35%, PC@4 67%, Joint@4 32%.
Final audit evidence is in `reports/20261022_c_subset100_complete/`.
This is still a C-only subset pilot, not evidence of improvement over a baseline.

On 2026-10-22 the user reduced the run to 100 groups. The full-generation service
and VBench weight-preparation service were stopped. Do not restart either of them
for this pilot. Their earlier snapshots below remain historical evidence.

The active service is `wan-said-c-subset100-20261022.service`, defined in
`systemd/`. Its output is `runs/pilot_v2/c_subset100_20261022/`.
This is C only: exactly 100 prompts and 100 videos from VideoPhy-2, no VBench.
The 100 prompts are sampled without replacement using seed 20261022 from the
original 590-job plan, sorted into original plan order. The frozen
`subset_lock.json` includes every selected job ID and the full plan checksum.
Neither existing videos nor scores affect selection. Matching already-generated
C videos are checksum/geometry verified and hardlinked, with reuse provenance.

Four independent GPU workers use the unchanged full generation protocol below,
not smoke settings. Their combined expected coverage is 100; all 100 must pass
identity, checksum and geometry checks before official AutoEval SA/PC scoring.
The summaries report SA mean, PC mean, SA@4, PC@4 and Joint@4 with denominator 100.
Missing, duplicate or invalid scores block publication of the subset table.
The table and JSON explicitly retain `evaluation_scope=subset_pilot`,
`full_benchmark_denominator=590`, `formal_benchmark_complete=false`,
`main_table_ready=false` and `publication_eligible=false`.
`subset_table_ready=true` means only this 100-video pilot completed.
It is not a complete VideoPhy-2 or VBench benchmark result.

At the measured approximately 515 seconds/video on each of four RTX 3090s,
generation is approximately 3.6 hours before reuse, followed by scoring.
Budget roughly 3-4 hours under similar load, not a guaranteed finish time.
The completed training and sealed 500-video dataset test split are unchanged.

```bash
systemctl --user status wan-said-c-subset100-20261022.service
# Resume only the requested pilot, not the historical full queue:
systemctl --user start wan-said-c-subset100-20261022.service
```

Live status is `runs/pilot_v2/c_subset100_20261022/status.json`.
Final pilot scores appear in that run's `videophy2/scores/summary.json` and CSV
only after complete, valid scoring. Code, selection, worker logs and test output
are snapshotted in `reports/20261022_c_subset100/`; MP4s and weights remain local.

## Historical Full Queue

The completed training/validation queue did not execute the requested benchmarks.
This separate queue corrects that omission. Training diagnostics are not benchmark
scores, and a successful one-video evaluator probe is not a 590-video result.

## Frozen Inputs

- Adapter: `runs/pilot_v2/c_validation_20261021/C/checkpoint.pt`.
- Adapter SHA256: `cadf8a40f5c51885fffc8affc37cd8953313ee1f93332e785791e93af97610af`.
- Full original plans: `runs/pilot_v2/c_validation_20261021/plans/`.
- Output: `runs/pilot_v2/c_official_20261022/`.
- VideoPhy-2: all 590 prompts, one fixed seed each; not early VideoPhy's 344 prompts.
- VBench: 944 unique prompts, five fixed seeds each, 4720 videos.
- Generation: 81 frames, 480x832, 16 FPS, official Wan UniPC 50 steps,
  CFG 6, shift 8, original prompts, fixed official negative prompt, no extension.
- No teacher, projector, grounding, or disk feature cache during generation.
- At least 50 GiB free disk is retained. No old datasets are removed.

Four independent generators read and validate the same complete official plan.
Worker `r` generates `jobs[r::4]` on its own GPU. Sharding does not change the
denominator, seeds, prompts, or sampling protocol. A shared file lock serializes
contract creation; worker progress and video records have disjoint names.

## Execution And Recovery

The stopped historical user service is `wan-said-c-official-20261022.service`; its complete
machine-specific definition is versioned in `systemd/`. It survives closing this
chat, but is not configured to start after reboot. Do not restart it for the
current 100-video request. Inspect its historical state with:

```bash
systemctl --user status wan-said-c-official-20261022.service
```

The queue generates all VideoPhy-2 videos, verifies each record, checksum and
geometry, executes the pinned official SA and PC scripts, and aggregates
Joint@4 with one-to-one video/caption matching. Missing or invalid scores block
the summary; they never reduce 590. It then generates all VBench videos and
executes all 16 official dimensions and the official weighted aggregation.

If VBench's verified scorer environment is not yet ready, status explicitly says
`vbench_waiting_for_scorer_environment`. It never substitutes a custom metric or
claims that queued scoring has run. Dependency/weight preparation runs separately
as `wan-said-vbench-setup-20261022.service`, without allocating GPU memory.

Generation resumes only from matching recorded videos. An unrecorded MP4, changed
checksum, source/adapter identity change, failed worker, or insufficient free space
is an error. A failed worker terminates and reaps its siblings. Scorer failures
preserve logs and stop the queue. Previously successful scoring is revalidated
with `--aggregate-only`, including raw-result and environment hashes.

## Evaluator Integration

The official VideoPhy-2 checkpoint's nested configs specify Transformers 4.28.1.
The initial 4.31 environment failed because upstream MplugOwlConfig copied the
bound `_rope_scaling_validation` method and could not serialize its configuration.
The isolated scorer environment now uses 4.28.1; neither the official scorer
source nor the scoring model weights were edited. Both SA and PC successfully
scored the first real C video. Failed and successful probe logs are retained
separately; probe scores must not appear in a benchmark comparison table.

VBench has its own environment, pinned dependencies, source revisions, model
checksums and local tokenizer files. Its weight readiness lock is published only
after all required model files and evaluator imports are verified. Installing or
downloading dependencies is not evidence of real VBench scoring.

The original RAFT Dropbox link is unreachable from this server. Its named
`raft-things.pth` is obtained through HF mirror from
`dc-ai/vbench_pretrained_models@d6ddad44430e861088ae49409dac073d04dc7e12`.
SHA256 `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`
also matches `sbalani/raft-things@3cda7e7fe51ad8f594c2824a6d6ee13b7f61afdc`.
This is corroboration between two mirrors, not a publisher-provided checksum;
the environment records that provenance limitation explicitly.

## Interpretation

This uses the complete official benchmark protocol on an exploratory C adapter.
Human review remains pending, so outputs retain `publication_eligible=false`.
There is no completed matched A/B comparison and no proof yet that SAID improves
generation. The 500 sealed dataset test videos remain unused.

`reports/20261022_official_start/` is a point-in-time audit snapshot, not live
progress. `reports/20261022_official_running/` preserves the subsequent snapshot
with the final queue implementation and environment preparation fixes.
Live progress is the run's `status.json`; final results appear only in
`videophy2/scores/summary.json` and `vbench/scores/summary.json` after success.
Use `collect_c` with a fresh report directory to preserve subsequent audit logs;
checkpoints, scoring weights and generated MP4s are never copied into Git.
