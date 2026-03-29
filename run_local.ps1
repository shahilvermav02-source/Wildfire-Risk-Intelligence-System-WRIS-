Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e . --no-build-isolation

python scripts/download_data.py
python scripts/train_model.py
python scripts/run_api.py
