#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase reality_memory_train
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
RESTREAM_NPROC="${RESTREAM_WORLD_SIZE:-}"
if [[ -z "$RESTREAM_NPROC" ]]; then
  RESTREAM_NPROC="$("$RESTREAM_PYTHON" - "$@" <<'PY'
import argparse
from pathlib import Path
import yaml
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', type=Path, default=Path('configs/reality_memory_r0.yaml'))
args, _ = parser.parse_known_args()
print(int(yaml.safe_load(args.config.read_text())['train']['world_size']))
PY
)"
fi
"$RESTREAM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$RESTREAM_NPROC" train_reality_memory.py "$@"
