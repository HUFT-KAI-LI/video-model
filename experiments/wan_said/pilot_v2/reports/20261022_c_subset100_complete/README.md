# C 100-Video Pilot: Completed

The requested pilot completed on 2026-10-22 at approximately 14:48 CST.
The four-GPU service started at 11:10:35 CST, for about 3 hours 37 minutes of
wall time including generation, verification and official AutoEval scoring.
The service exited successfully; this was evaluation, not additional training.

## Results

| Metric | C, 100-video subset |
| --- | ---: |
| Semantic adherence (SA), mean / 5 | 3.16 |
| Physical commonsense (PC), mean / 5 | 3.78 |
| SA >= 4 | 35 / 100 (35%) |
| PC >= 4 | 67 / 100 (67%) |
| Joint@4: SA >= 4 and PC >= 4 | 32 / 100 (32%) |

All 100 selected videos were generated and verified. Both pinned official
scoring tasks exited with code 0. Each score file has 100 valid integer-valued
ratings in 1..5. One-to-one video/caption matching found no missing, duplicate,
unexpected or invalid scores; the denominator was not reduced.

The selection seed is 20261022. The frozen 100-job subset was drawn from the
unchanged original 590-job VideoPhy-2 plan without using scores or existing
video availability. Four matching earlier C videos were verified and hardlinked;
96 were newly generated. Generation used the unchanged 81-frame, 480x832,
16 FPS, 50-step official UniPC protocol, CFG 6 and shift 8, on four RTX 3090s.
No teacher inference or disk feature caching was added.

## Audit And Limits

- `run/subset_lock.json`: all selected IDs, original plan checksum and seed.
- `run/run_lock.json`: C adapter, source and generation-protocol identities.
- `run/videophy2/videos/`: generation lock, per-worker logs and per-video records.
- `run/videophy2/scores/`: raw SA/PC CSVs, official scorer logs, execution status,
  summary JSON with paired ratings, and explicitly subset-labelled summary CSV.
- `snapshot.json`: test exit codes, service state and snapshot time.
- `source_sha256.json` and `artifact_sha256.json`: source and evidence checksums.

This is an exploratory C-only **subset pilot**, not the complete 590-prompt
VideoPhy-2 benchmark and not early VideoPhy. VBench was not run. The summary
retains `formal_benchmark_complete=false`, `main_table_ready=false`,
`subset_table_ready=true` and `publication_eligible=false`.
There is no matched FM-only A baseline or A/B/C comparison here, so these numbers
do not establish that SAID improves video generation. Human review remains
pending. The sealed 500-video dataset test split was not used.

Only code and audit evidence are uploaded to GitHub. Model weights, the training
checkpoint, generated MP4s and raw datasets remain local. The earlier
`20261022_c_subset100` report remains the historical startup snapshot.
