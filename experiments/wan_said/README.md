# Wan2.1-1.3B + SAID, Experiment 1

This is an isolated research experiment, not a validated improvement to Wan.
The original dirty `video-model` worktree and LongLive/Reality Memory logic are unchanged.
Branch: `codex/wan-said-exp1`. New worktree:
`/data/lk/深度学习/video-model-said-exp1`.

## Current Evidence

- Fifteen unit tests pass, including three independent gradient objectives and split isolation.
  An additional opt-in integration test checks QC/splitting on 40 real downloaded OpenVid clips.
- A/B/C each completed a matched 5-step real Wan smoke test on the old 32-training-video subset.
- FM updates LoRA in all 30 blocks. H15 distillation updates blocks 1-15 and the student projector,
  not blocks 16-30 or the frozen target branch. Checkpoint prediction roundtrip error is zero.
- Twelve normal-path videos were generated from four prompts and one seed, with no Teacher,
  Mask, Grounding network, or student projector loaded at inference.
- These are engineering checks, not an A/B/C efficacy comparison. The old Grounding validation
  retrieval is approximately chance (1/6); prompt sensitivity alone does not establish semantics.
  The 256x256 smoke generation is heavily stylized/distorted. No VBench or T2V-CompBench scores
  are reported. Pilot training and Unsaid are deliberately not started.
- Tracked evidence: `reports/`. Actual checkpoints, videos and logs: `runs/` (git-ignored).

## Expanded Dataset, Under 100 GB

Source: `nkp37/OpenVid-1M`, via **https://hf-mirror.com**, pinned revision
`d8a63bd22989c80b5734ec2bb989f4e1b61a5807`.
Selected: **5,000** distinct original sources, 4,000 Panda clips and 1,000 Pixabay clips.
Raw videos total **13,270,085,773 bytes**; caption CSV is **856,907,992 bytes**.
Do not confuse a fixed selection with completed downloads: consult
`datasets/openvid/pipeline_status.json` and `download_progress.json` for live status.

ZIP members are read with HTTP Range and the established `remotezip` library. Servers that
ignore Range are rejected; there is no full-archive fallback. CRC is checked during extraction,
and each local video receives a SHA256 record. The original ZIPs are never stored. Raw videos
are capped at 40 GB, total data at 100 GB, with a conservative feature-cache budget check before
selection. Existing finished files are checksum-verified before being skipped on retry.

The subset is seeded and source-unique before downloading. Parent YouTube IDs and Pixabay
source IDs are parsed conservatively; unknown naming formats are excluded. After the complete
selection has downloaded, geometry/duration/decode QC, exact hashes, normalized captions and
8-frame perceptual signatures are checked. Near duplicates are removed before a family-stratified
**80/10/10 train/val/test** split. Target counts are about **4,000/500/500**; actual counts can be
lower after logged QC/deduplication. More than 5% QC rejection stops preparation for inspection.

Training, Grounding pretraining, teacher targets and negative pools use **train only**.
Validation is for diagnostics/model selection, not optimizer inputs. Test remains sealed until
the protocol and checkpoint are fixed. All three manifests have a SHA256 lock, rechecked by
every dataset loader; cache entries are bound to the split lock and sample identity.

Limitations: this is not an official OpenVid benchmark split; it does not establish independence
between channels/episodes or absence from Wan pretraining. pHash is not a semantic deduplication
guarantee. Caption paraphrases and clip-level text/video accuracy need manual review. The 4 MB
clip cap and selected archive IDs introduce sampling bias. The dataset card specifies research/
non-commercial intent in addition to CC-BY-4.0 and underlying source licenses; retain provenance.

The old 32/6/6 manifests in `manifests/` are **smoke only** and preserve all prior training exposure
in train. New data are never mixed into these manifests. Use `config_openvid.json` and
`manifests_openvid/` for the expanded pilot.

## Implementation

Frozen original umT5 uses padding-aware pooling only for the auxiliary text projector. The normal
512-token Wan condition is retained with padding zeroed. VideoMAEv2 ViT-small (distilled from
the K710 giant, 384 dimensions) is frozen. Its raw tokens and frozen pooled features are cached,
not the trainable projection outputs. Its official weight SHA256 is in `upstream_lock.json`.
It is **not** assumed naturally aligned with umT5.

Grounding pretraining uses a real multi-sample batch of train-only cached features. The joint gate
sees `[text, video, text * video, abs(text - video)]` and selects 256 semantic dimensions. Every
positive and negative pair uses the same pair-conditioned scoring rule. Bidirectional InfoNCE
excludes duplicate captions/sources. A sparse penalty and keep-budget anchor discourage collapse.
Full/random/text-shuffled diagnostic controls are evaluated with heldout validation features.

This implementation is **staged**, not end-to-end joint training: train Grounding, freeze it, then
optimize DiT LoRA and the H15 student projection. B has an independently pretrained unmasked
Grounding projection with the same pretraining batch/step budget as C. Grounding cost is disclosed
separately. A does not incur this extra cost. DiT-stage C loss is `FM + 0.05 * Distill`, with **no**
Grounding loss in that optimizer. Sparse regularization is not an Unsaid objective.

The FM objective, timestep distribution, noise construction, target and weighting are imported
directly from pinned DiffSynth-Studio `FlowMatchSFTLoss` and `FlowMatchScheduler("Wan")`.
No local replacement formula is used. The backbone and native forward also come from that
unmodified pinned source, not the old LongLive implementation. LoRA rank 8 targets q/k/v/o in
all attention blocks. H15 is `blocks[14]`; capture only records the tensor inside the hook and
pools outside the checkpoint region. Hooks are removed before backward recomputation.

Teacher targets are stop-gradient normalized features. Zero hard masks yield finite zero
distillation supervision. BF16, non-reentrant gradient checkpointing and fixed-size, unpadded
video grids are used. Noise-level diagnostics are logged; five smoke steps happen not to contain
a low-noise sample, which is recorded as an empty bucket rather than an invented metric.
Clip-level pooling does not demonstrate object binding or fine action alignment.

## Environment

The base environment is read-only `/data/lk/深度学习/.envs/video-model`: torch 2.5.1+cu124,
transformers 4.49.0, peft 0.14.0. The experiment has its own `.venv` with system-site packages
and only the additions in `requirements-extra.txt`. Full tested versions are in
`reports/requirements-frozen.txt`. Four RTX 3090 24 GB are available; GPU 3 handles text caching
and generation, GPU 1 VAE caching, GPU 2 video features/Grounding, GPU 0/1/2 A/B/C DiT runs.
These are independent GPU jobs, not an assumption that device memories add together.

Vendor source revisions and model hashes are pinned in `upstream_lock.json`. Fresh setup:

```bash
cd /data/lk/深度学习/video-model-said-exp1
BASE=/data/lk/深度学习/.envs/video-model/bin/python
$BASE -m venv --system-site-packages experiments/wan_said/.venv
PY=experiments/wan_said/.venv/bin/python
$PY -m pip install -r experiments/wan_said/requirements-extra.txt
git -c http.proxy= -c https.proxy= clone https://github.com/modelscope/DiffSynth-Studio.git experiments/wan_said/vendor/DiffSynth-Studio
git -C experiments/wan_said/vendor/DiffSynth-Studio checkout b34c83793f783b8bf0c7e82142e06dda01f98d27
git -c http.proxy= -c https.proxy= clone https://github.com/OpenGVLab/VideoMAEv2.git experiments/wan_said/vendor/VideoMAEv2
git -C experiments/wan_said/vendor/VideoMAEv2 checkout 29eab1e8a588d1b3ec0cdec7b03a86cca491b74b
```

The Wan weights are shared read-only at the absolute path in the configs. For a fresh host, adjust
that path only after verifying the hashes in `reports/phase0.json`. Do not use LongLive weights.
Teacher weight mirror URL:
`https://hf-mirror.com/OpenGVLab/VideoMAE2/resolve/706cc172d65ebd4dedbee3f9c0183a93df9fa125/distill/vit_s_k710_dl_from_giant.pth`.
Expected SHA256: `24fb71687fa3671b8387cadfbcbab0f72af695692e93cf1ecc82caa888626172`.

## Reproduction

Run from the new worktree root. Always use fresh output directories.

```bash
PY=experiments/wan_said/.venv/bin/python
$PY -m unittest discover -s experiments/wan_said/tests -v
$PY -m experiments.wan_said.mirror_download --repo nkp37/OpenVid-1M --revision d8a63bd22989c80b5734ec2bb989f4e1b61a5807 --file data/train/OpenVid-1M.csv --output experiments/wan_said/datasets/openvid/metadata/OpenVid-1M.csv
$PY -m experiments.wan_said.openvid_subset plan --parts 20 45 60 70 84 100 110 130 150 170 --samples 5000
$PY -m experiments.wan_said.data_pipeline --workers 32
```

The one-shot data pipeline stops after QC and immutable manifests. It does not silently start
model training or repeatedly retune on test. The durable systemd user job is
`wan-said-openvid-20261021.service`. Progress/log paths:

- `datasets/openvid/pipeline_status.json`
- `datasets/openvid/download_progress.json`
- `datasets/openvid/pipeline.log`
- `datasets/openvid/quality_rejections.json`
- `manifests_openvid/split_lock.json` (exists only after preparation succeeds)

```bash
systemctl --user status wan-said-openvid-20261021.service
```

After `raw_data_ready`, prepare the expanded train/val features (never test). The following
three cache commands may run independently on the configured GPUs:

```bash
$PY -m experiments.wan_said.cache_features --kind text --config experiments/wan_said/config_openvid.json
$PY -m experiments.wan_said.cache_features --kind vae --config experiments/wan_said/config_openvid.json
$PY -m experiments.wan_said.cache_features --kind teacher --config experiments/wan_said/config_openvid.json
$PY -m experiments.wan_said.train_grounding --mode full --config experiments/wan_said/config_openvid.json --output experiments/wan_said/runs/openvid_ground_full
$PY -m experiments.wan_said.train_grounding --mode said --config experiments/wan_said/config_openvid.json --output experiments/wan_said/runs/openvid_ground_said
```

First inspect validation retrieval, shuffled-text/video sensitivity, random/full controls and gate
collapse. If grounding does not generalize, fix it before spending a long DiT pilot. For the
first real-model test on the new data, run matched 10-step A/B/C smoke jobs:

```bash
$PY -m experiments.wan_said.train --group A --config experiments/wan_said/config_openvid.json --steps 10 --gradient-audit --output experiments/wan_said/runs/openvid_smoke_A
$PY -m experiments.wan_said.train --group B --config experiments/wan_said/config_openvid.json --steps 10 --gradient-audit --grounding experiments/wan_said/runs/openvid_ground_full/grounding.pt --output experiments/wan_said/runs/openvid_smoke_B
$PY -m experiments.wan_said.train --group C --config experiments/wan_said/config_openvid.json --steps 10 --gradient-audit --grounding experiments/wan_said/runs/openvid_ground_said/grounding.pt --output experiments/wan_said/runs/openvid_smoke_C
```

Only after these checks pass, use fresh runs with the same three commands and `--steps 300`.
This is a pilot, not evidence sufficient for a paper. The frozen prompt suite is
`compositional_prompts.json`. Native evaluation command (no extra inference modules):

```bash
$PY -m experiments.wan_said.generate --config experiments/wan_said/config_openvid.json --A experiments/wan_said/runs/openvid_smoke_A/checkpoint.pt --B experiments/wan_said/runs/openvid_smoke_B/checkpoint.pt --C experiments/wan_said/runs/openvid_smoke_C/checkpoint.pt --output experiments/wan_said/runs/openvid_generation
```

Generation outputs include MP4s, an A/B/C side-by-side, a contact sheet, native-path metadata and
an intentionally unscored human-review CSV. Attribute/action claims require inspecting the actual
videos. No metrics should be filled from expectation or teacher-feature cosine alone.
