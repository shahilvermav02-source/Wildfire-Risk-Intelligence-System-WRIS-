# Wildfire Risk Intelligence System (WRIS)

A complete mini ML + API + GUI project for wildfire risk prediction.

## ✅ Stable Run Design (Codespaces-friendly)

- No mandatory editable install for running.
- Uses stdlib-only runtime and `PYTHONPATH=src` execution.
- One-command scripts for Linux/macOS (`run_local.sh`) and Windows (`run_local.ps1`).
- GitHub Actions CI runs tests without packaging/install friction.

---

## 0) Quick Colab Run

Use `COLAB.md` for one-cell execution.

## 1) Clone from GitHub

```bash
git clone https://github.com/shahilvermav02-source/Wildfire-Risk-Intelligence-System-WRIS-.git
cd Wildfire-Risk-Intelligence-System-WRIS-
```

If using branch from Codex:

```bash
git checkout codex/build-wildfire-risk-intelligence-system
```

---

## 2) Quick run (recommended)

### Linux / macOS / GitHub Codespaces

```bash
chmod +x run_local.sh
./run_local.sh
```

### Windows PowerShell

```powershell
.\run_local.ps1
```

Open GUI:
- `http://127.0.0.1:8000/gui`

If model artifact is missing, API now auto-downloads data and trains on first prediction request.

---

## 3) Manual run (step-by-step)

```bash
export PYTHONPATH="$(pwd)/src"
python scripts/download_data.py
python scripts/train_model.py
python scripts/run_api.py
```

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
