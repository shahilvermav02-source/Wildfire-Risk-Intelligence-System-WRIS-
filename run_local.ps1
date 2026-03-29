Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$env:PYTHONPATH = "$PWD/src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

python scripts/download_data.py
python scripts/train_model.py
python scripts/run_api.py
