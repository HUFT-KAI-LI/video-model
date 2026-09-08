#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase eval
"$RESTREAM_PYTHON" eval_reanchor.py "$@"
