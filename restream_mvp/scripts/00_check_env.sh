#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase environment
nvidia-smi || true
command -v nvcc >/dev/null && nvcc --version || true
df -h "$RESTREAM_ROOT"
free -h
"$RESTREAM_PYTHON" - <<'PY'
import sys, torch
print('python:',sys.version)
print('torch:',torch.__version__)
print('cuda:',torch.cuda.is_available(), 'count:',torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i,torch.cuda.get_device_name(i))
PY
