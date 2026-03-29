#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e . --no-build-isolation

python scripts/download_data.py
python scripts/train_model.py
python scripts/run_api.py
