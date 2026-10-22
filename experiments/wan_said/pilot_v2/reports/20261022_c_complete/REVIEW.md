# C Pilot Completion: October 22, 2026

This is a completed exploratory training/held-out diagnostic run, not a
completed official generation benchmark or publication-ready efficacy claim.
All times below are Asia/Shanghai (UTC+8).

## Completion And Integrity

- C training completed at 06:17 on October 22: 999 DDP optimizer updates,
  local batch 1/global batch 4, exactly 3995 presentations of 3995 unique train
  videos. The final partial batch did not duplicate supervised examples.
- Training-loop time: 32869.63 seconds (9.13 hours). The latest 20 updates
  averaged 26.12 seconds, faster than the approximately 47.6 seconds observed
  near the beginning. No resumed or silently skipped samples were required.
- Validation completed at 08:21: all 498 registered validation videos scored,
  in 7378.29 seconds (2.05 hours). The denominator was not reduced.
- Service exited successfully (exit code 0). Final LoRA hashes agree across all
  four ranks. The native FM/SAID algorithms, frozen split and preprocessing are
  unchanged. Frozen Teacher/Grounding received no optimization updates in C.
- The 500 sealed test videos were not used; no persistent feature cache was
  created. Representative human caption/crop review is still pending, so the
  checkpoint remains explicitly exploratory and not publication eligible.

## Held-Out Diagnostics

| Metric | Result |
| --- | --- |
| Original base model native weighted FM loss | 0.10683255985727062 |
| C native weighted FM loss | 0.1063958930251858 |
| Paired C minus base FM difference | -0.00043666683208481614 |
| Paired difference standard error | 0.00004053513222520145 |
| Relative decrease in mean FM loss | 0.408739% |
| C Said cosine distillation loss | 0.3425354018029439 |
| Grounding Said video-to-text R@1 | 24.8996% |
| Grounding Said text-to-video R@1 | 22.4900% |
| Retrieval chance R@1 (1/498) | 0.2008% |

The paired FM values use the same fixed per-video noise/timestep for base and C,
with one draw per validation video. This is a small improvement in the registered
diagnostic, not a video quality score and not evidence that Said adds benefit
beyond ordinary FM LoRA tuning. A matched fully trained FM-only arm is still
needed to isolate that effect. No convergence or best-checkpoint claim is made.

## Semantic Selection Remains Unproven

Using the same trained Grounding projections, changing the mask to all ones
produced v2t/t2v R@1 of 25.5020%/22.6908%. Shuffling only the text supplied to
the gate produced 24.8996%/22.6908%; the retrieval text itself stays correctly
paired in this diagnostic. The Said mask did not outperform the all-channel
diagnostic, and shuffling gate text barely changed retrieval performance.

These are warning signs of weak demonstrated text-conditioned selection, not a
proof that the mask ignores text completely: the soft-mask prompt-change L1 was
0.00420367 and the existing nondegeneracy gate passed. The report correctly keeps
`semantic_selectivity_proven=false`. The all-channel diagnostic above is NOT an
independently trained B/Full experiment.

## What Is Not Complete

- No official VideoPhy-2 or VBench scorer was executed and no official generation
  quality scores are available. `complete=true` refers only to the registered
  train/validation/benchmark-planning queue with an empty generation suite list.
- Full official benchmark generation remains deferred under the existing budget
  decision; planning prompt files is not executing those benchmarks.
- No matched full-training A/B comparison or multi-seed efficacy study is present.
- The adapter/optimizer checkpoint remains local (about 87.85 MiB). This snapshot
  uploads logs, diagnostics, tests, config and source hashes, not checkpoint
  weights, raw videos or cached features.

See `run/C/report.json`, `run/validation/summary.json`, `run/status.json`, and
the original per-step/per-video records for machine-readable evidence.
