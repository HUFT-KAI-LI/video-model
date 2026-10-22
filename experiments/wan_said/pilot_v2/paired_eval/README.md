# Matched Base/C Evaluation

This module completes the requested paired evaluation without changing Wan,
SAID, the training algorithm or the pinned official benchmark source code.

## Fixed Scope

- Models: untouched Wan2.1-T2V-1.3B (`base`, no adapter) and the existing C adapter.
- VideoPhy-2: the identical frozen 100 prompts and seeds already evaluated on C.
  C's completed 100-video result is verified and reused; base generates 100 videos.
- VBench: 100 official annotated prompt groups, all 16 dimensions covered, five
  unchanged official-plan seeds per group: 500 generated videos per model.
- Within each benchmark, base and C share exactly the same prompt/seed jobs.
  The two benchmarks retain their respective official prompt suites. Physics
  prompts are not relabeled as VBench's annotated semantic prompts.
- Total: 1200 evaluated videos across both models/suites, 1100 new generations.
  At the previous approximately 515 s/video per RTX 3090, four-GPU generation
  alone is approximately 39.3 hours, followed by scoring/verification overhead.
- Generation remains 81 frames, 480x832, 16 FPS, 50-step native UniPC, CFG 6,
  shift 8, original prompts and fixed native negative prompt. No teacher or
  disk feature cache. A 50 GiB free-disk reserve remains enforced.

The VBench seed-20261022 selection first covers at least five prompt groups per
dimension, prioritizing rare dimensions, then fills the 100 groups from the same
random shuffle. It never uses generated quality or scores for selection. The
selected metadata rows, including auxiliary detection targets, are unchanged
copies of the pinned official metadata. All five original seeds are retained.

## Execution

`systemd/wan-said-paired100-20261022.service` runs the pipeline on all four GPUs:

1. Verify and reuse C's existing VideoPhy-2 result.
2. Generate and score base VideoPhy-2.
3. Generate and score C VBench.
4. Generate and score base VBench.
5. Reverify all matching jobs, video checksums, native geometry, generation
   contracts, official raw score hashes, scoring model locks and summary hashes.
6. Produce the complete comparison and archive/publish audit evidence to GitHub.

Four independent native generators each occupy one GPU. For VBench scoring,
sixteen unmodified official dimension invocations are scheduled over four GPUs,
one task per visible physical GPU. Failed tasks stop the queue and terminate
their siblings. Rerunning failed VBench scoring archives previous outputs rather
than mixing result files or deleting evidence. A successful scorer is rechecked
using its result hashes before aggregation is reused.

Weight preparation is independent, uses the HF domestic mirror, verifies file
hashes and preserves source provenance. Detectron2 must include its CUDA
operators even when GPUs are hidden during compilation; this does not allocate
training GPU memory. A verified scorer environment is published only after
model acquisition and all sixteen evaluator imports pass. Real scoring failure
is never treated as a successful environment probe or benchmark result.
The separate `wan-said-paired100-evidence-20261022.service` waits for readiness
and copies the verified environment, dependency freeze, wrapper and preparation
journal into the run's `evidence/` directory. No model weight bytes are copied.
Download failures use bounded service retries; readiness is never inferred just
from a running preparation process.

Live output is `runs/pilot_v2/paired100_20261022/status.json`. The service survives
closing the chat but is not enabled after reboot. Resume only after checking
that no duplicate GPU job is running:

```bash
systemctl --user status wan-said-paired100-20261022.service
systemctl --user start wan-said-paired100-20261022.service
```

Source, model, plan or subset changes reject resumption; they require an explicit
new run or an audited migration rather than silently combining old/new results.

## Final Results

`comparison/comparison.json`, CSV and Markdown are produced only after all four
model/suite results pass verification. They contain:

- SA mean, PC mean, SA@4, PC@4, Joint@4 and paired base/C deltas; deterministic
  10,000-resample paired bootstrap 95% intervals over the 100 VideoPhy-2 pairs.
- All sixteen VBench raw scores and upstream normalized/weighted dimensions.
- Official VBench quality, semantic and total scores using the pinned upstream
  aggregation functions, not a custom sixteen-dimension mean.
- Scheduled/scored counts for every dimension and each model, with exclusions.

The official color evaluator excludes clips where the target object is not
detected. The subset evaluator retains that official conditional aggregate but
explicitly records every excluded clip and the unchanged scheduled denominator.
It does not zero-impute or claim strict video coverage. Color score differences
may reflect different detection coverage; review those counts before comparing.
Other missing/duplicate/invalid scores or absent dimensions block the table.

These are matched 100-prompt **subset** results, not full benchmark leaderboard
scores. Base versus C measures combined fine-tuning effects; isolating SAID still
requires an FM-only fine-tuned A control. C remains exploratory and human review
is pending. The sealed 500-video dataset test split remains unused.

At completion, `--publish` creates a fresh audit snapshot under `pilot_v2/reports`,
runs all experiment regression suites and commits/pushes only that evidence
directory to `codex/wan-said-c-eval`. Unrelated staged changes block automatic
publication. Videos, model weights and checkpoints are never added to Git.
If upload fails, local completed scores remain available and status explicitly
records the publication error; it does not fabricate an uploaded report.
