#!/usr/bin/env bash
set -euo pipefail
RESTREAM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RESTREAM_ROOT"
mkdir -p logs
RESTREAM_PYTHON="$RESTREAM_ROOT/.venv/bin/python"
if [[ ! -x "$RESTREAM_PYTHON" ]]; then RESTREAM_PYTHON=python3; fi
export PYTHONPATH="$RESTREAM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
log_phase() {
  exec > >(tee -a "logs/${1}_$(date +%Y%m%d_%H%M%S).log") 2>&1
}
