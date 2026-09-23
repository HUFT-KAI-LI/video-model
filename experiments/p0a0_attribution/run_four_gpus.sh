#!/usr/bin/env bash
# Run from repository root; never start a second copy against active GPU workers.
set -euo pipefail
run_dir=${1:-experiments/p0a0_attribution/runs/a0_20260923}
python_bin=restream_mvp/.venv/bin/python
mkdir -p "$run_dir"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
"$python_bin" experiments/p0a0_attribution/verify_assets.py --run "$run_dir"
pids=()
for gpu_id in 0 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u experiments/p0a0_attribution/generate.py \
    --run "$run_dir" --model wan > "$run_dir/wan_gpu${gpu_id}.log" 2>&1 &
  pids+=("$!")
done
(
  CUDA_VISIBLE_DEVICES=1 "$python_bin" -u experiments/p0a0_attribution/generate.py \
    --run "$run_dir" --model longlive > "$run_dir/longlive_gpu1.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 "$python_bin" -u experiments/p0a0_attribution/generate.py \
    --run "$run_dir" --model wan > "$run_dir/wan_gpu1.log" 2>&1
) &
pids+=("$!")
result=0
for worker_pid in "${pids[@]}"; do
  wait "$worker_pid" || result=1
done
if [[ "$result" != 0 ]]; then
  echo 'A worker failed. Preserve its output and inspect logs before retrying.' >&2
  exit 1
fi
"$python_bin" -u experiments/p0a0_attribution/finalize.py --run "$run_dir"
