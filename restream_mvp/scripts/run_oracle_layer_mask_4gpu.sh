#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing oracle optimization from a dirty worktree" >&2
  exit 2
fi

commit="$(git rev-parse --short=12 HEAD)"
root="validation/oracle_layer_masks/screen_${commit}"
mkdir -p "$root"

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
    .venv/bin/python scripts/run_oracle_layer_mask.py \
    --reviewed --oracle-approved --gpu 0 --shard "$gpu" --shards 4 \
    --output "$root/shard${gpu}.json" \
    --video-root "$root/shard${gpu}_videos" \
    --cache-dir "$root/shard${gpu}_cache" \
    >"$root/shard${gpu}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "At least one oracle shard failed; inspect $root/shard*.log" >&2
  exit 1
fi

.venv/bin/python scripts/analyze_oracle_layer_mask.py \
  --results "$root/shard0.json" "$root/shard1.json" "$root/shard2.json" "$root/shard3.json" \
  --output "$root/analysis.json"

echo "Oracle analysis: $root/analysis.json"
