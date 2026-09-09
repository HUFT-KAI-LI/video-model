#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase reality_memory_prepare
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$RESTREAM_PYTHON" scripts/download_reality_encoder.py
"$RESTREAM_PYTHON" scripts/filter_continuous_shots.py
"$RESTREAM_PYTHON" scripts/build_reality_manifest.py
"$RESTREAM_PYTHON" scripts/check_reality_encoder.py
"$RESTREAM_PYTHON" scripts/cache_reality_features.py
"$RESTREAM_PYTHON" scripts/check_reality_ready.py
"$RESTREAM_PYTHON" -m unittest discover -s tests -v
printf '%s\n' 'Reality Memory R0 prepared. Stop for review; no video training or evaluation launched.'
