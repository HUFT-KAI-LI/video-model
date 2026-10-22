# C Official Benchmark Queue

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

The installed user service is `wan-said-c-official-20261022.service`; its complete
machine-specific definition is versioned in `systemd/`. It survives closing this
chat, but is not configured to start after reboot. Inspect or restart it with:

```bash
systemctl --user status wan-said-c-official-20261022.service
systemctl --user start wan-said-c-official-20261022.service
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
