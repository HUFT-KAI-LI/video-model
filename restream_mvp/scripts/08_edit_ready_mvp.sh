#!/usr/bin/env bash
# Edit-Ready Video Generation MVP.
#
#   smoke      two prompts, two target chunks, one GPU; writes summary_smoke.json
#   all-chunks same, but persists every AR chunk boundary (heavy-cache path)
#   main       8 prompts x 2 target chunks sharded across $SHARDS GPUs (default 4)
#
# Experiment A = scripts/check_edit_cache_replay.py (Gate A replay exactness)
# Experiment B = scripts/run_chunk_edit.py        (Gates B/C/D local edit + cost)
#
# One process per GPU; shards are whole prompts so two shards never regenerate
# the same base video or write the same cache file.  Wall-clock timing is part of
# the reported cost, so shards are run one experiment at a time, not stacked.
source "$(dirname "$0")/common.sh"
mkdir -p logs
LOG_STAMP="$(date +%Y%m%d_%H%M%S)"

MODE="${1:-smoke}"
shift || true
SHARDS="${SHARDS:-4}"
CONFIG="${CONFIG:-configs/edit_ready_mvp.yaml}"
OUT="${OUT:-validation/edit_ready_mvp}"
CACHE="${CACHE:-outputs/edit_ready_mvp/cache}"

run_replay_shards() {
  local tag="$1" output_prefix="$2" cache_dir="$3" extra=("${@:4}")
  local pids=()
  for ((gpu = 0; gpu < SHARDS; gpu++)); do
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
      "$RESTREAM_PYTHON" scripts/check_edit_cache_replay.py \
      --reviewed --gpu 0 --config "$CONFIG" \
      --shard "$gpu" --shards "$SHARDS" \
      --output "${output_prefix}_shard${gpu}.json" --cache-dir "$cache_dir" \
      "${extra[@]}" > "logs/edit_ready_${tag}_shard${gpu}_${LOG_STAMP}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || return 1; done
}

run_edit_shards() {
  local tag="$1" output_prefix="$2" cache_dir="$3" video_root="$4" extra=("${@:5}")
  local pids=()
  for ((gpu = 0; gpu < SHARDS; gpu++)); do
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
      "$RESTREAM_PYTHON" scripts/run_chunk_edit.py \
      --reviewed --gpu 0 --config "$CONFIG" \
      --shard "$gpu" --shards "$SHARDS" \
      --output "${output_prefix}_shard${gpu}.json" \
      --cache-dir "$cache_dir" --video-root "$video_root" \
      "${extra[@]}" > "logs/edit_ready_${tag}_shard${gpu}_${LOG_STAMP}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || return 1; done
}

case "$MODE" in
  smoke)
    # Writes into the git-ignored scratch directory first so the working tree
    # stays clean while the jobs run (and the recorded provenance stays sealable),
    # then copies the artifacts into validation/edit_ready_mvp/.
    SHARDS=1
    SMOKE_OUT="$OUT/rerun_smoke"
    mkdir -p "$SMOKE_OUT" "$CACHE/replay_smoke" "$CACHE/edit_smoke"
    run_replay_shards replay_smoke "$SMOKE_OUT/replay_smoke" "$CACHE/replay_smoke" \
      --cases 2 --targets 0 1
    run_edit_shards edit_smoke "$SMOKE_OUT/local_edit_smoke" "$CACHE/edit_smoke" \
      "$SMOKE_OUT/videos_smoke" --cases 2 --targets 0 1
    "$RESTREAM_PYTHON" scripts/summarize_edit_ready_mvp.py \
      --config "$CONFIG" --replay "$SMOKE_OUT/replay_smoke_shard0.json" \
      --edit "$SMOKE_OUT/local_edit_smoke_shard0.json" --output "$SMOKE_OUT/summary_smoke.json"
    cp "$SMOKE_OUT/replay_smoke_shard0.json" "$SMOKE_OUT/local_edit_smoke_shard0.json" \
       "$SMOKE_OUT/summary_smoke.json" "$OUT/"
    rm -rf "$OUT/videos_smoke"
    cp -r "$SMOKE_OUT/videos_smoke" "$OUT/videos_smoke"
    ;;
  all-chunks)
    SHARDS=1
    mkdir -p "$OUT" "$CACHE/all_chunks"
    run_replay_shards replay_all_chunks "$OUT/replay_all_chunks" "$CACHE/all_chunks" \
      --cases 1 --targets 0 1 4 --all-chunks
    ;;
  main)
    mkdir -p "$OUT" "$CACHE/replay_main" "$CACHE/edit_main" "$OUT/videos"
    run_replay_shards replay_main "$OUT/replay_main" "$CACHE/replay_main"
    run_edit_shards edit_main "$OUT/local_edit_main" "$CACHE/edit_main" "$OUT/videos" --dino
    "$RESTREAM_PYTHON" scripts/summarize_edit_ready_mvp.py \
      --config "$CONFIG" --output "$OUT/summary.json"
    ;;
  main-edit)
    # Re-run only Experiment B and re-aggregate (Experiment A outputs are reused).
    mkdir -p "$OUT" "$CACHE/edit_main" "$OUT/videos"
    run_edit_shards "${EDIT_TAG:-edit_main}" "${EDIT_PREFIX:-$OUT/local_edit_main}" \
      "$CACHE/edit_main" "$OUT/videos" --dino "${@}"
    "$RESTREAM_PYTHON" scripts/summarize_edit_ready_mvp.py \
      --config "$CONFIG" --output "$OUT/summary.json"
    ;;
  rerun)
    # Sealed rerun on a clean tree: every artifact goes to a git-ignored scratch
    # directory so `git status --porcelain` stays empty while the jobs run and the
    # recorded provenance can honestly say git_dirty=false.
    RERUN_OUT="$OUT/rerun"
    mkdir -p "$RERUN_OUT" "$CACHE/rerun_replay" "$CACHE/rerun_edit" "$RERUN_OUT/videos"
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false \
      "$RESTREAM_PYTHON" scripts/check_edit_streaming_equivalence.py --reviewed --gpu 0 \
      --output "$RERUN_OUT/streaming_equivalence.json" \
      > "logs/edit_ready_equivalence_${LOG_STAMP}.log" 2>&1
    saved_shards="$SHARDS"
    SHARDS=1
    run_replay_shards clean_all_chunks "$RERUN_OUT/replay_all_chunks" "$CACHE/rerun_all_chunks" \
      --cases 1 --targets 0 1 4 --all-chunks
    SHARDS="$saved_shards"
    run_replay_shards clean_replay "$RERUN_OUT/replay_main" "$CACHE/rerun_replay"
    run_edit_shards clean_edit "$RERUN_OUT/local_edit_main" "$CACHE/rerun_edit" \
      "$RERUN_OUT/videos" --dino
    run_edit_shards clean_edit_seed2 "$RERUN_OUT/local_edit_seed2" "$CACHE/rerun_edit" \
      "$RERUN_OUT/videos" --dino --seed-stride 1
    "$RESTREAM_PYTHON" scripts/summarize_edit_ready_mvp.py --config "$CONFIG" \
      --output "$RERUN_OUT/summary.json"
    ;;
  *)
    echo "usage: $0 {smoke|all-chunks|main|main-edit|rerun}" >&2
    exit 2
    ;;
esac
