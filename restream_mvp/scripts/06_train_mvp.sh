#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase train
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
"$RESTREAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 train_reanchor.py "$@"
