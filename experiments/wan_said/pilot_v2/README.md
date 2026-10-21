# Experiment 1: ViT-Base / Full-Timeline 5K Pilot

This is an isolated adaptation of `experiments/wan_said`, not a replacement of
the old 17-frame / ViT-Small experiment. It does not edit LongLive, Reality
Memory, the shared alignment implementation, or upstream FM code.

## Scope And Status

- The latest user instruction takes precedence over the attached 64K acquisition
  target: use the already downloaded 5K pilot and prioritize algorithm adaptation.
- 5,000 original videos, 4,993 after deduplication: train 3,995 / val 498 / test 500.
  Original source/hash/caption-group isolation and split membership are preserved.
- This size-capped pilot is biased toward files <=4 MB. It is NOT the official
  VideoREPA 64K set and is not presented as a paper main-table training set.
- The 64K downloader remains paused. Existing videos and old experiments remain.
- No text, teacher-token, pooled-feature, VAE-latent, or full-dataset RAM cache is
  created. Frozen encoders recompute current-batch features online. Model weights,
  adapters, optimizer checkpoints, logs, previews, and generated evaluation videos
  still consume storage. `feature_cache_enabled=true` is rejected.
- Test videos cannot enter the online training/Grounding provider. Previews and
  diagnostic retrieval use train/val only. Sealed test metadata is used solely for
  split integrity and benchmark-caption leakage checks.
- Short real-model tests are engineering evidence, not evidence of better generation
  quality or semantic selection. The C pilot is now being launched in stages;
  see `EVALUATION.md` and the dated progress snapshots, not the older smoke report,
  for its actual status. Official scoring has not yet produced real scores.

## Algorithm Kept Intact

Shared code is imported, not copied or changed:

- `alignment.py`: Grounding, JointGate, masked pooling, Projector, cosine loss,
  duplicate-negative exclusion, and non-reentrant H15 capture.
- `runtime.py`: strict native Wan weight loading, LoRA injection, native forward,
  frozen umT5/VAE, and adapter serialization.
- `train.py`: shared gradient inspection only. Training loss directly calls
  pinned DiffSynth `FlowMatchSFTLoss` with its unmodified Wan training scheduler.

For pooled text `t` and pooled ViT-Base video `v`, the original Grounding module
computes `zt=normalize(Pt(t))`, `zv=Pv(v)`, and the joint channel mask
`m=sigmoid(MLP([zt,zv,zt*zv,abs(zt-zv)]))`. Said targets are
`stop_gradient(normalize(m*zv))`; Full targets use `normalize(zv)`.
Grounding pretraining retains the shared symmetric contrastive objective,
same-caption/source negative exclusions, and the existing sparsity/budget term.
Student supervision remains video-level cosine distillation, not token-level SAID.

| Arm | Objective | Grounding |
| --- | --- | --- |
| A | Native FM | None |
| B | Native FM + 0.05 * Full cosine distillation | Fresh Full checkpoint |
| C | Native FM + 0.05 * Said cosine distillation | Fresh Said checkpoint |

Distillation uses `dit.blocks[14]`: pooled H15 `1536 -> 256`.
Text remains `4096 -> 256`; video changes `384 -> 768 -> 256` by replacing only
the frozen encoder/input dimension. B and C use the exact same frozen Teacher and
preprocessing, but independently pretrained Full/Said Grounding modules, as in
the shared implementation. No Unsaid, TRD, new loss, or layer ablation is added.

## New Input Contract

1. Verify source SHA256; decode all frame PTS, then select 49 nearest frames
   spanning the first through last decoded PTS. The old middle-window field is
   removed. Repeated source frames and maximum sampling gaps are recorded.
2. Apply one aspect-aware center crop, resize to `49 x 480 x 720`, and send that
   exact content to the native Wan VAE (`1 x 16 x 13 x 60 x 90` latent).
3. Teacher consumes those SAME cropped pixels, normalizes by ImageNet statistics,
   removes frame 0, and downsamples to `48 x 160 x 240`.
4. Load VideoREPA's pinned VideoMAEv2 ViT-Base K710-distilled backbone strictly.
   Only the four documented classifier/fc_norm keys are removed. Final block
   raw tokens (no classifier/fc_norm) have shape `1 x 3600 x 768`; mean pooling
   feeds the existing Grounding objective. Raw tokens remain available per batch.
5. Interpolate canonical original position embeddings spatially with bicubic and
   temporally with trilinear interpolation. Temporal grids are parameterized:
   Wan 17/49/81 frames -> Teacher 16/48/80 -> 1200/3600/6000 tokens.

Endpoint coverage is not proof of retaining every action. Center crop can remove
subjects; sparse sampling can miss short events. `audit.py` writes full-range
keyframe sheets and a named human review CSV. `review.py` records decisions and
reasons. Any rejection blocks formal training until a corrected/filtered dataset
is separately frozen; this code never silently removes samples or changes splits.
An approved representative review is not a corpus-wide semantic certification.
The original caption string is retained, but native umT5 conditioning still has
Wan's 512-token limit. The audit records token-length quantiles, the number of
over-limit captions, and sampled-caption truncation flags. Full source text does
not imply every token reaches the model; no silent caption rewriting is used.

## Execution And Resume

Default four-GPU placement is student/LoRA on GPU 0, VAE on GPU 1, Teacher on GPU 2,
and umT5 on GPU 3. Grounding runs on GPU 0 before student training. This is model
placement, not four-way DDP; student effective batch size is one video per step.
No full epoch is implied by a short smoke test.

All arms use the same LoRA initialization, epoch-seeded data permutations, and
per-step noise/timestep seeds. `execution.py` scopes the documented deterministic
FlashAttention 2 backward option to this new training entry point, restoring the
upstream module reference afterward. FM/native forward math is unchanged. The
initial audit exposed nondeterministic FA2 backward behavior; this was corrected
before the final checked runs. Other execution backends require revalidation.

Resume restores the optimizer, adapters, head, step history, and Grounding states.
Contracts bind teacher/source/preprocessing, split-lock hash, base weight hashes,
LoRA configuration, optimizer protocol, Grounding hash, arm, and smoke status.
Changed protocols are rejected. Old ViT-Small checkpoints/caches are not reused.
Fresh adapters and optimizer states are saved periodically and at completion.

## Commands

Run from the repository root with the existing isolated environment. Upstream
commits and the official Teacher checkpoint are locked in
`../protocol_v2/sources_lock.json`; shared native sources in `../upstream_lock.json`.
Both lock files remain authoritative. `config.json` contains this machine's Wan
weight directory; adjust paths in a separate configuration on another machine.
The shared environment uses `requirements-extra.txt` on the existing training
environment; dependency installation does not upgrade that environment.

```bash
PY=experiments/wan_said/.venv/bin/python
$PY -m experiments.wan_said.pilot_v2.manifests
$PY -m experiments.wan_said.pilot_v2.audit --output experiments/wan_said/runs/pilot_v2/audit_new --count 8 --teacher-check
$PY -m experiments.wan_said.pilot_v2.smoke --output experiments/wan_said/runs/pilot_v2/smoke_new --ground-steps 3 --train-steps 2
$PY -m experiments.wan_said.pilot_v2.verify_runs --smoke-root experiments/wan_said/runs/pilot_v2/smoke_new --output experiments/wan_said/runs/pilot_v2/resume_new
$PY -m unittest discover -s experiments/wan_said/pilot_v2/tests -v
```

Formal runs require an actual named human review, not an auto-generated approval:

```bash
$PY -m experiments.wan_said.pilot_v2.review --sheet REVIEWED.csv --audit AUDIT.json --output APPROVAL.json
$PY -m experiments.wan_said.pilot_v2.grounding --mode full --steps 1000 --output FULL_DIR
$PY -m experiments.wan_said.pilot_v2.grounding --mode said --steps 1000 --output SAID_DIR
$PY -m experiments.wan_said.pilot_v2.train --group A --steps 3995 --approved-review APPROVAL.json --output A_DIR
$PY -m experiments.wan_said.pilot_v2.train --group B --steps 3995 --grounding FULL_DIR/grounding.pt --approved-review APPROVAL.json --output B_DIR
$PY -m experiments.wan_said.pilot_v2.train --group C --steps 3995 --grounding SAID_DIR/grounding.pt --approved-review APPROVAL.json --output C_DIR
```

Step counts above are examples, not an assertion that 1,000 Grounding steps are
sufficient. B/C reject near-random/unvalidated Grounding unless `--smoke` is
explicitly supplied. Grounding reports held-out retrieval versus chance, random
mask and shuffled-text controls, mask statistics, and prompt sensitivity; it does
not automatically declare semantic selectivity. Choose training budgets using
validation only. For resume, pass `--resume .../checkpoint.pt` (or Grounding's
`grounding.pt`); `--steps` means total steps, including those already completed.

## Benchmark Interfaces

`benchmarks.py` locks original prompt files, seeds, per-prompt output mappings,
source commits, and caption overlap checks. Formal inference uses the pinned
official Wan UniPC solver, 81 frames, 480x832, 16 FPS, 50 steps, CFG 6, shift 8,
the source-defined negative prompt, and no prompt extension. Teacher, Grounding,
and student projection head are absent during generation. Output hashes and
decoded geometry/frame counts are checked on resume. Smoke adapters cannot be
scored as formal models, and missing videos cannot reduce the denominator.

| Suite | Pinned Protocol |
| --- | --- |
| T2V-CompBench | 1,400 original prompts; all seven categories x 200 |
| VBench | 946 metadata entries / 944 unique prompts; five videos each = 4,720; all 16 dimensions |
| VideoPhy-2 | 590 original test prompts; AutoEval SA/PC 1-5; primary Joint@4 |
| VideoPhy (separate, not run here) | 344 test prompts; VideoCon-Physics binary SA/PC |

The 344-prompt early VideoPhy and 590-prompt VideoPhy-2 are different benchmarks,
not alternative regular sizes of one benchmark. Do not directly compare their
scores, including early VideoPhy results reported by VideoREPA. Such a comparison
requires a separate early-VideoPhy reproduction. Do not truncate VideoPhy-2.
The pinned alternate upsampled CSV has 591 rows and is not the default protocol.
Original captions in `videophy2.csv` remain fixed. `--accept-videophy-590` is now
only a backward-compatible no-op, not a scientific approval gate.
The pinned T2V source also has two prompt/metadata wording discrepancies: motion
row 195 (`lifed` / `lifted`) and action row 159 (`a eagle` / `an eagle`). Both source
files are hash-locked and preserved unchanged, with the differences recorded in
the suite audit. These are not silently "fixed" in generated prompts.

```bash
$PY -m experiments.wan_said.pilot_v2.benchmarks --output PLAN_DIR
$PY -m experiments.wan_said.pilot_v2.inference --jobs PLAN_DIR/t2v_compbench.jsonl --output PREVIEW_DIR --group C --checkpoint C_DIR/checkpoint.pt --smoke --limit 1
$PY -m experiments.wan_said.pilot_v2.inference --jobs PLAN_DIR/vbench.jsonl --output VVIDEOS --group C --checkpoint C_DIR/checkpoint.pt
$PY -m experiments.wan_said.pilot_v2.score --suite vbench --jobs PLAN_DIR/vbench.jsonl --videos VVIDEOS --output VSCORES --plan-only
```

Scoring recipes delegate to pinned official entry points, not CLIP proxies. They
are **not yet integration-validated**: separate official scorer environments and
large evaluator weights remain to be installed, hash-locked, and tested. For
execution, `--environments` takes a JSON file with interpreter paths and an explicit
`weights` list of `{path, sha256}` entries. Supported keys: `llava_python`,
`detector_python`, `dot_python`, `vbench_python`, `videophy_python`,
`llava_checkpoint`, and `videophy_checkpoint`. Detection/depth/tracking dependencies
must also follow their official repo layout. Offline flags prevent accidental HF
weight downloads; other backend setup remains the operator's responsibility.
Do not interpret recipe creation as successful metric execution. `score.py` now
automatically invokes strict aggregation after successful official scoring:
VideoPhy-2 identity-joined SA/PC means and Joint@4; VBench all 16 raw dimensions,
official normalization, quality, semantic and total scores. It writes auditable
`summary.json` and a comparison-ready `summary.csv` only on complete valid results.
`--aggregate-only` rechecks generation/weight/script/result hashes before reuse.
Model installation and complete real benchmark runs remain outstanding. Details,
including conditional upstream exclusions, are in `EVALUATION.md`.

## Review Map

- Input/Teacher: `video.py`, `teacher.py`, `config.py`, `manifests.py`.
- Algorithm wiring: `online.py`, `grounding.py`, `train.py`, `execution.py`.
- Validation: `audit.py`, `review.py`, `validate.py`, `smoke.py`, `verify_runs.py`, `tests/`.
- Evaluation: `benchmarks.py`, `inference.py`, `score.py`, `aggregate.py`.
- C workflow/progress: `run_c.py`, `collect_c.py`, `EVALUATION.md`.
- Actual executed commands, machine-readable checks, limitations: `reports/`.

Raw MP4s, model weights, optimizer checkpoints, full manifests containing local
data paths, and old feature caches are not committed. Reports preserve the split
lock and hashes; the original parent manifest files remain in the existing project.
