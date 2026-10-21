# Four-GPU Execution

This directory changes execution, not the shared FM/SAID algorithms. No disk
feature cache is created; original videos and frozen train/validation manifests
remain unchanged. The 500 sealed test videos are never loaded for training.

## Grounding

Four spawned encoder workers each own a local frozen T5 and VideoMAEv2. They
encode disjoint shards of the original eight-sample batch. The coordinator
restores the original order and calls the existing Grounding loss with all eight
contrastive negatives and the unchanged optimizer. Resume retains the serial
checkpoint, optimizer, sample sequence, and global batch. Floating-point
encoder results can differ slightly across GPU hardware/kernel placement.

The October 21 run resumes saved step 850; the original checkpoint is retained
as `grounding.serial850.pt`. Initial held-out pooled features live only in RAM.
GPU memory is not deliberately filled: this stage does not need student DiT/VAE.

## C Fine-Tuning

Use `torch.distributed.run --standalone --nproc_per_node=4 -m
experiments.wan_said.pilot_v2.parallel.train`. Every GPU owns a student and frozen
online encoders, not a different stage of one serial model. DDP averages gradients
across four distinct training videos per update. The original native
`FlowMatchSFTLoss`, H15 feature capture, projector, frozen Grounding target, and
`FM + 0.05 * Said` objective are reused.

**Execution protocol change:** local batch 1, global batch 4, learning rate still
1e-4. A single pass through 3995 videos is 999 optimizer updates, not 3995
four-video updates. The last update averages exactly three real examples;
the fourth rank participates with zero loss weight. No padded sample is counted
twice. Sample order and per-sample noise seeds match the original shuffled epoch.
This changes optimizer dynamics relative to serial batch 1; it is explicitly
recorded and is not claimed to be byte-identical to the serial run. A/B comparison
runs must use this same execution/batch protocol.

Resume requires the same world size, batch, identity, Grounding SHA, and review
status. Checkpoints retain the inference/validation-compatible LoRA/projector
format. Rank-zero-only writes avoid checkpoint races. The initial checkpoint is
written after one update, then every 50 updates, and at completion.

## Memory And Other Users

Each process has an allocator ceiling of min(20 GiB, currently free VRAM minus
2 GiB). This is a ceiling, not a promised occupancy or a dummy allocation.
When available budget is below 19 GiB, T5 is moved back to CPU after text encoding
so student backward and VAE encoding can fit. Only current-batch features are
retained. Other users' processes are never stopped. In particular, GPU 1 already
had another user's approximately 4.7 GiB job on October 21, 2026.

The process fails on inadequate memory, nonfinite features/gradients, encoder
worker errors, or a changed resume contract. It never silently skips samples.
Gradient audit checks FM reaches all 30 blocks and Said only reaches blocks 1-15.
Grounding validation and human-review/publication gates are unchanged.

Validation remains the registered full 498-video paired base/C diagnostic stage.
Full VideoPhy-2/VBench generation remains deferred under the existing budget
decision; a training loss is not a generation-quality benchmark score.
