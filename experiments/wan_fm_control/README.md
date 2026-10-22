# Matched FM-Only Control (A)

This independent experiment adds the missing traditional flow-matching control
without changing shared training algorithms, the C trainer, or the running
Base/C evaluation's frozen Python sources. It is **LoRA fine-tuning**, matching C,
not full-parameter training. The native DiffSynth `FlowMatchSFTLoss` is called
unchanged. No teacher, projector, Grounding, H15 hook or auxiliary loss is used.

## Registered Protocol

- Start from the original Wan2.1-T2V-1.3B and the identical initial LoRA as C.
- Reuse C's frozen resolved config: rank/alpha 8, q/k/v/o, LR 1e-4, AdamW,
  weight decay 0.01, clip 1, BF16, gradient checkpointing, seed 42.
- 3995 train clips, one epoch, four DDP ranks, local batch 1/global batch 4,
  999 updates. Last batch has three supervised clips and zero-weight padding.
- Identical shuffled examples, per-sample noise seeds, diffusion timesteps and
  crop/timeline metadata are checked against C at completion.
- Reuse shared text/VAE encoders, deterministic attention and sampling. Only
  current-batch features reside in RAM; no disk feature caches are created.
- 498 held-out validation clips use the same fixed noise seeds as C. The sealed
  500 dataset test clips are not used. Validation FM is not generation quality.
- Preserve the existing exact 100 VideoPhy-2 prompt/seed lock. Use official SA,
  PC and Joint@4; write an interim three-arm comparison as soon as scored.
- Preserve the existing exact 100 VBench annotated prompt groups and all five
  seeds (500 videos/model), all 16 official dimensions and official aggregation.
- Shared verified summaries enforce identical base weights, native 81-frame
  480x832/50-step UniPC generation, prompts/seeds and scorer weight locks.
- Color conditional exclusions remain explicit per model; no missing-score
  imputation. Subset results never claim full official benchmark completion.

## Durable Execution And Resource Handoff

`python -m experiments.wan_fm_control.run --output RUN --publish` freezes source
and input checksums, then temporarily stops **only**
`wan-said-paired100-20261022.service`. Completed generation records/videos remain
untouched. Up to four in-flight videos may need regeneration after the handoff.
Other users' jobs are never stopped.

A real native-geometry four-GPU preflight compares one update resumed to two
against uninterrupted two-update training and audits all 30 FM gradient blocks.
Smoke checkpoints are separate and cannot be used as final benchmark adapters.
The full A run starts from the original model, not the smoke or C adapter.

Order: preflight -> A full epoch -> 498 validation -> A VideoPhy-2 generation and
official scoring -> interim Base/A/C comparison and audit upload -> restore
Base/C VBench queue -> await its completion -> A VBench -> complete comparison
and audit upload. Four generators/scorers share the existing helpers. A cap of
20 GiB allocated per training process is a ceiling, not artificial memory fill.
At least 50 GiB free disk is retained for generation/model setup.

The service template includes `ExecStopPost` recovery so a failed/terminated A
run restores the previously active unfinished Base/C queue. A completed queue
is never restarted by recovery. No auto-reboot enablement or automatic training
retry is configured. To resume after inspecting a failure, restart this service;
matching checkpoints and verified completed videos are reused. If an external
kill leaves an empty output directory without a checkpoint, preserve/rename it
and restart with a fresh stage output rather than fabricating a resume.

## Outputs And Interpretation

`status.json`, `run_lock.json`, `preflight_proof.json`, `A/report.json`,
`matched_training.json`, `validation/summary.json`, `videophy2_comparison/`, and
finally `comparison/` provide auditable intermediate/final state. Raw models,
MP4s and datasets stay local; small logs, score CSVs, locks, receipts and test
logs are archived under `wan_said/pilot_v2/reports/` and published to the existing
GitHub branch. No score is reported before the official scorer succeeds.

The final table reports Base, FM-only A and FM+SAID C, with A-Base, C-A and C-Base
deltas. VideoPhy-2 includes paired bootstrap intervals (10000 resamples, fixed
seed). C-A is the relevant alignment comparison under this single training seed;
it is not proof of universal significance. Human caption/crop review remains
pending, so all adapters/results remain exploratory and publication-ineligible.

Run focused CPU tests with:

```bash
experiments/wan_said/.venv/bin/python -m unittest discover -s experiments/wan_fm_control/tests -v
```
