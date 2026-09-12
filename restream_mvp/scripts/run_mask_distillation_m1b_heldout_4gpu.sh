#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing M1-B held-out run from dirty tree" >&2; exit 2
fi
commit="$(git rev-parse --short=12 HEAD)"
root="validation/mask_distillation_m1b/heldout_${commit}"
mkdir -p "$root"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    .venv/bin/python scripts/run_mask_distillation_heldout.py \
    --reviewed --heldout-approved --gpu 0 --shard "$gpu" --shards 4 \
    --manifest validation/mask_distillation_m1b_heldout_manifest.json \
    --plan configs/mask_distillation_m1b_plan.json \
    --controllers validation/mask_controllers_m1b.pt \
    --teacher-dataset validation/mask_teacher_dataset_m1b.json \
    --training-report validation/mask_controller_training_m1b.json \
    --output "$root/shard${gpu}.json" --cache-dir "$root/shard${gpu}_cache" \
    --video-root "$root/shard${gpu}_videos" >"$root/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ "$failed" -ne 0 ]]; then echo "At least one M1-B held-out shard failed" >&2; exit 1; fi
.venv/bin/python scripts/analyze_mask_distillation_heldout.py \
  --plan configs/mask_distillation_m1b_plan.json \
  --results "$root/shard0.json" "$root/shard1.json" "$root/shard2.json" "$root/shard3.json" \
  --output "$root/analysis.json"
echo "M1-B analysis: $root/analysis.json"
