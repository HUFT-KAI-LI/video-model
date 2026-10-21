# C Pilot And Corrected Evaluation Protocol

## Benchmark Identity

| Benchmark | Test prompts | Evaluator | Main joint condition |
| --- | --- | --- | --- |
| VideoPhy / VideoPhy-1 | 344 | VideoCon-Physics, binary SA/PC | SA=1 and PC=1 |
| VideoPhy-2 | 590 | VideoPhy-2-AutoEval, integer SA/PC 1-5 | SA>=4 and PC>=4 |

These are different benchmarks. The earlier task's 344-prompt VideoPhy-2 wording
and the copied VideoREPA README's 344-video comment are not the current protocol.
The pinned `VideoREPA/evaluation/VIDEOPHY2/videophy2.csv` supplies all 590 original
captions. The alternate 591-row upsampled CSV is not substituted. The official
Hritikbansal/videophy `VIDEOPHY2/README.md` confirms the rating scales and joint
threshold; the top-level README identifies VideoPhy-1 as ICLR 2025 and VideoPhy-2
as ICLR 2026. AutoEval scores are not human leaderboard scores.

Early VideoPhy scores in another paper cannot be compared directly with these
590-prompt VideoPhy-2 scores. Early VideoPhy reproduction is a separate task.

## VideoPhy-2 Aggregation

`Joint@4 = count(SA_i >= 4 AND PC_i >= 4) / N`, where N is the full frozen plan
denominator, 590. Report SA mean, PC mean, SA@4, PC@4 and Joint@4; rates are
fractions, not percentages. The last three are different statistics.

SA and PC CSVs are joined by resolved absolute video path, not CSV row position.
Each row must match its original caption and appear exactly once in each task.
Ratings must be finite integers 1-5. The upstream parser's default 0 is treated
as a failure, not a real rating. Missing, duplicate, unexpected, wrong-caption
or invalid rows withhold the complete scores. The audit retains N and every
failure. No failed-video denominator reduction or score imputation is used.

## Official VBench Aggregation

Pinned VBench revision: `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`.
The standard metadata has 946 rows but 944 unique video-generation prompts.
Five seed-indexed videos per prompt gives 4,720 videos, with every metadata row
and its dimension membership preserved. A dimension is evaluated on its own
official prompt subset, not on all 4,720 videos.

The adapter verifies the upstream Git blob identity and SHA256 of
`scripts/cal_final_score.py`, `scripts/constant.py` and the metadata. It then
directly calls the unchanged upstream functions:

1. `get_nomalized_score` (upstream spelling): dimension-specific Min/Max
   normalization and weights. No clipping. Dynamic degree has weight 0.5;
   other dimensions have weight 1.
2. `get_quality_score`: official seven quality dimensions with their weights.
3. `get_semantic_score`: official nine semantic dimensions.
4. `get_final_score`: quality:semantic = 4:1.

Sparse vendor checkouts must include the pinned `/scripts/` directory, in addition
to `/vbench/` and `/evaluate.py`. The current local checkout was expanded to
include these official files; no vendored source was edited.

Use each official dimension's aggregate output, not a new per-video mean:
some official aggregates are frame weighted. Do not use a 16-dimension simple
mean or the upstream submission helper's missing-dimension zero fallback.
Raw and normalized weighted dimension values, quality, semantic and total are
written to the comparison CSV with model/checkpoint/plan identity.

Coverage checks compare every result's full video path and multiplicity to the
corresponding official metadata subset. All 16 dimensions must be present once;
NaN/Inf, duplicate dimensions/files, unexpected paths and missing scores block
the complete total. VBench `color.py` intentionally omits videos with no detected
object, and temporal flickering can skip exceptions. Such omissions are recorded
and the total is withheld under our strict complete-coverage policy. This is
conservative: it does not silently change official formulas or claim an
incomplete run as a complete benchmark. Resolving exclusions requires a separately
audited scorer run, not filling zero or reducing a denominator.

## C Training And Validation

The current user request is a preliminary 5K pilot, not a 64K paper experiment.
Training uses all 3,995 train videos once (3,995 optimizer steps, batch 1), fixed
seed 42, 49x480x720 full-timeline samples, native FM + 0.05 Said. Existing shared
FM/SAID/Grounding math and original projects are unchanged. Encoders remain
frozen and feature caches remain disabled. Four GPUs retain the existing
placement: student/Grounding 0, VAE 1, ViT-Base 2, umT5 3, not four-way DDP.

Grounding first receives 1,000 registered contrastive optimizer steps, batch 8,
from the entire 3,995-video training pool. Sampling is with replacement across
steps, so 8,000 presentations are not 8,000 unique videos. Its fixed held-out
64-video validation cohort must pass the existing nondegeneracy and bidirectional
R@1-above-chance gate. No checkpoint is chosen using benchmark/test scores.
Failure at step 1,000 stops the pipeline rather than bypassing the gate or
silently extending training until a favorable validation score appears.

The representative human review remains pending. `--exploratory` permits this
requested preliminary pilot without inventing approval, while preserving the
Grounding validation gate. It is distinct from `--smoke`; checkpoints are tagged
exploratory and publication-ineligible, and inference requires explicit matching
`--exploratory`. This status propagates to generated-video locks and score tables.
The default non-exploratory human-review requirement is unchanged. A rejected
review is not bypassed: if supplied, the review must still pass validation.

After C finishes one full train pass, `validate.py` uses **all 498 validation
videos**, never the 500 sealed test videos. It compares base vs C native weighted
FM loss with the same per-video noise seed/timestep, reports paired FM delta and
its standard error, C's Said loss, and held-out Grounding retrieval/diagnostics.
There is one fixed noise/timestep draw per validation video. These are training
diagnostics, not substitutes for VBench or VideoPhy-2 quality scores. Pooled
validation features are held temporarily in RAM for retrieval, never disk-cached;
raw tokens, latents and contexts are recomputed online.

Next, benchmark generation uses unchanged native UniPC, 81 frames, 480x832,
16 FPS, 50 steps, CFG 6, shift 8, original prompts and fixed plan seeds. It is
resumable and checks existing file hashes and decoded geometry. Smoke adapters
remain ineligible. VideoPhy-2 and VBench produce 590 and 4,720 videos respectively.
The pipeline scores each suite only when its verified environment/weight lock is
available. Missing evaluator installation leaves an explicit pending stage,
not fabricated scores. T2V-CompBench's prior 1,400-job plan is retained but is not
part of this C runner's two requested corrected benchmark suites.

## Execution And Audit

The first Grounding service started on 2026-10-21 at 17:24:03 Asia/Shanghai:
`wan-said-c-grounding-20261021.service`. The C pipeline waits for its successful
exit and validated final checkpoint before loading the student. Run root:
`experiments/wan_said/runs/pilot_v2/c_validation_20261021`.

Example commands, prefixed by the existing experiment venv Python:

```bash
-m experiments.wan_said.pilot_v2.run_c --output experiments/wan_said/runs/pilot_v2/c_validation_20261021 --grounding-service wan-said-c-grounding-20261021 --ground-steps 1000 --exploratory
-m experiments.wan_said.pilot_v2.score --suite videophy2 --jobs RUN/plans/videophy2.jsonl --videos RUN/generation/videophy2 --output RUN/scores/videophy2 --environments RUN/scorers/videophy2_environment.json
-m experiments.wan_said.pilot_v2.score --suite vbench --jobs RUN/plans/vbench.jsonl --videos RUN/generation/vbench --output RUN/scores/vbench --environments RUN/scorers/vbench_environment.json --aggregate-only
-m experiments.wan_said.pilot_v2.collect_c --run RUN --output NEW_REPORT_SNAPSHOT
```

`run_lock.json` freezes source hashes, dataset, config, step budget and benchmark
selection. Code or data changes during the run cause a stop at the next stage.
`status.json` and phase logs distinguish waiting/running/failed/completed states.
Reports are point-in-time snapshots, not future completion claims. Checkpoints,
raw videos and evaluator weights are never uploaded to GitHub. Later full logs
can be collected with `collect_c.py` into a fresh report directory for review.

The official VideoPhy-2 evaluator uses repository `videophysics/videophy_2_auto`,
revision `ca38370e1dcc79c99c7017dd8755d0a087299afc`. `scorer_setup.py` downloads via
`https://hf-mirror.com`, concurrency 2, checks upstream LFS SHA256 and sizes, and
reserves 50 GiB free disk space. It downloads about 14.31 GB of checkpoint and
configuration files, not the API's larger repository historical storage figure.
Only small resumable-download metadata is retained, not duplicate model blobs or
training feature caches. The first proxy-routed request failed TLS; direct
domestic-mirror access was selected without disabling certificate validation.

The evaluator has a separate environment (Transformers 4.31.0, PEFT 0.4.0,
Accelerate 0.21.0, Decord 0.6.0); the training environment is unchanged. Imports
have been checked, but real evaluator inference awaits downloaded weights and
generated videos. VBench's separate full model/dependency stack is still pending.

Tests use synthetic result fixtures to check joins, coverage, failure handling,
pipeline wiring and exact equality with official VBench aggregation. Their
fixture scores must never be described as scores from the trained C model.
