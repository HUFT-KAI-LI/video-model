#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase smoke
"$RESTREAM_PYTHON" scripts/smoke_infer.py "$@"
