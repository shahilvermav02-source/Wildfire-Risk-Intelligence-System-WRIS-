# Wildfire Risk Intelligence System (WRIS)

A complete mini ML + API + GUI project for wildfire risk prediction.

## ✅ What is fixed in this rebuild

- Runs from source without manual `PYTHONPATH` tweaks.
- Works after cloning from GitHub (local scripts bootstrap path automatically).
- Includes one-command run scripts for Linux/macOS and Windows PowerShell.
- Includes GitHub Actions CI + Docker publish workflow.

---

## 1) Clone from GitHub

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

If using branch from Codex:

```bash
git checkout codex/build-wildfire-risk-intelligence-system
```

---

## 2) Quick run (recommended)

### Linux / macOS

```bash
./run_local.sh
```

### Windows PowerShell

```powershell
./run_local.ps1
```

These scripts do all steps: create venv, install, download data, train model, run API.

---

## 3) Manual run (step-by-step)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . --no-build-isolation
python scripts/download_data.py
python scripts/train_model.py
python scripts/run_api.py
```

Open GUI:
- `http://127.0.0.1:8000/gui`

---

## 4) Push your Codex branch to GitHub

```bash
git add .
git commit -m "rebuild: stable run-from-github flow"
git push -u origin codex/build-wildfire-risk-intelligence-system
```

Then create PR to `main`.

---

## 5) Run from GitHub-built Docker image (GHCR)

After merging to `main`, workflow publishes image:

```bash
docker pull ghcr.io/<owner>/<repo>:main
docker run --rm -p 8000:8000 ghcr.io/<owner>/<repo>:main
```

Open:
- `http://127.0.0.1:8000/gui`

---

## 6) Dataset

- UCI Forest Fires:  
  `https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv`
- If blocked, system uses embedded fallback sample automatically.

