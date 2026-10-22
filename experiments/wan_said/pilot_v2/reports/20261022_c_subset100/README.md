# Requested C 100-Video Pilot

This snapshot records the scope reduction requested on 2026-10-22, not a
completed benchmark. The previous full-generation and VBench-preparation
services were stopped; the new 100-video service started at 11:10:35 CST.

- Scope: exploratory C adapter, 100 VideoPhy-2 prompts, one video per prompt.
- Fixed selection seed: 20261022, no replacement, independent of scores and
  existing generated videos. See `run/subset_lock.json` for all selected IDs.
- Original benchmark size: 590, retained explicitly in every subset report.
- Four matching previous C videos were verified and hardlinked. The remaining
  96 videos are queued across four GPUs. See `run/reuse_report.json`.
- Native generation settings are unchanged: 81 frames, 480x832, 16 FPS,
  50 official UniPC steps, CFG 6, shift 8. No teacher or disk feature cache.
- Official VideoPhy-2 AutoEval SA and PC run automatically after all 100
  videos pass identity, checksum and geometry verification. Aggregation reports
  SA mean, PC mean, SA@4, PC@4 and Joint@4, requiring all 100 paired valid scores.
- VBench is not queued. There are no completed 100-video scores at snapshot time.
- This is a subset pilot, not the complete VideoPhy-2 benchmark. The adapter is
  still exploratory; no matched A/B comparison or generation improvement claim
  is available. The sealed 500-video dataset test split is unchanged and unused.

The service survives closing the chat, but is not enabled after reboot.
Live progress is in `runs/pilot_v2/c_subset100_20261022/status.json`.
Final subset metrics will be in that run's `videophy2/scores/summary.json`.
The tracked snapshot is not a live view; only code and audit evidence are
uploaded to Git, never model weights, checkpoints, or generated MP4s.

Synthetic test results validate plumbing and aggregation, not model quality.
Test exit codes and service state are recorded in `snapshot.json`; source and
artifact checksums are preserved alongside it.
