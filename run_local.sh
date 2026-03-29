#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

python scripts/download_data.py
python scripts/train_model.py
python scripts/run_api.py
