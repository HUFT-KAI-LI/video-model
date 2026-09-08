#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase setup
if [[ ! -x .venv/bin/python ]]; then python3 -m venv --system-site-packages .venv; fi
.venv/bin/python -m pip install -r requirements-runtime.txt
.venv/bin/python -c 'import torch; print(torch.__version__); assert torch.__version__.startswith("2.5.")'
# Reuse this machine's CUDA 12.4 PyTorch and flash-attn; do not replace drivers.
