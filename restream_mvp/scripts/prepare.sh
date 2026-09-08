#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
log_phase prepare
bash scripts/00_check_env.sh
bash scripts/01_setup_env.sh
bash scripts/02_download_models.sh
"$RESTREAM_PYTHON" scripts/03_download_youku_subset.py --output data/raw/youku --max-gb 8 --max-items 1200
"$RESTREAM_PYTHON" scripts/04_build_manifest.py --raw data/raw/youku/raw_manifest.jsonl --output data
"$RESTREAM_PYTHON" -m unittest discover -s tests -v
"$RESTREAM_PYTHON" scripts/check_ready.py
printf '%s\n' 'Preparation complete. Stop here for human review; no training or evaluation started.'
