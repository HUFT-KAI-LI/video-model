#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing M1-A teacher generation from a dirty worktree" >&2; exit 2
fi
commit="$(git rev-parse --short=12 HEAD)"
root="validation/mask_distillation/teachers_${commit}"
mkdir -p "$root"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    .venv/bin/python scripts/run_mask_distillation_teachers.py \
    --reviewed --teacher-approved --gpu 0 --shard "$gpu" --shards 4 \
    --output "$root/shard${gpu}.json" --cache-dir "$root/shard${gpu}_cache" \
    >"$root/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one M1-A teacher shard failed" >&2; exit 1
fi
echo "M1-A teachers: $root"
