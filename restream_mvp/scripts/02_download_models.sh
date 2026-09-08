#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase models
"$RESTREAM_PYTHON" scripts/download_models.py --model wan
"$RESTREAM_PYTHON" scripts/download_models.py --model longlive-modelscope
