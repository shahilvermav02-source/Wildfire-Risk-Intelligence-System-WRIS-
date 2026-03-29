# Wildfire Risk Intelligence System (WRIS)

WRIS is a mini end-to-end ML project for wildfire risk estimation.
It downloads the UCI Forest Fires dataset from the internet, trains a model, and serves predictions via an HTTP API + browser GUI.

## Architecture

- **Data Layer (`wris.data`)**: Downloads and loads wildfire CSV data.
- **Feature Layer (`wris.features`)**: Converts month/day and weather conditions to model-ready numeric vectors.
- **Model Layer (`wris.models`)**: Trains a weighted KNN regressor with simple cross-validation tuning and evaluates metrics.
- **Service Layer (`wris.services`)**: Loads saved model artifacts and computes risk score/level.
- **API + GUI Layer (`wris.api`)**: Standard-library HTTP server with `/health`, `/predict`, and GUI routes (`/`, `/gui`).

## Dataset

- Source: UCI Machine Learning Repository (Forest Fires)
- URL used by pipeline:  
  `https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv`
- If internet access is blocked, WRIS automatically uses an embedded fallback sample dataset so training and GUI still work.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . --no-build-isolation
```

## Run End-to-End

1. Download data
```bash
python scripts/download_data.py
```

2. Train model
```bash
python scripts/train_model.py
```

3. Run API + GUI server
```bash
python scripts/run_api.py
```

4. Open GUI
- `http://127.0.0.1:8000/` or `http://127.0.0.1:8000/gui`

## Example API Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "x": 7.4,
    "y": 4.1,
    "month": "aug",
    "day": "fri",
    "ffmc": 91.5,
    "dmc": 145.4,
    "dc": 678.2,
    "isi": 8.3,
    "temp": 29.1,
    "rh": 35,
    "wind": 3.6,
    "rain": 0.0
  }'
```

## Deploy with Docker (Local)

```bash
docker build -t wris:latest .
docker run --rm -p 8000:8000 wris:latest
```

Then open `http://127.0.0.1:8000/gui`.

## Deploy from GitHub (CI/CD)

This repo now includes GitHub Actions workflows:

- `.github/workflows/ci.yml`
  - Runs tests on every push and pull request.
- `.github/workflows/docker-ghcr.yml`
  - Builds Docker image and pushes to GHCR (`ghcr.io/<owner>/<repo>`) when pushing to `main`.

### Steps

1. Push project to GitHub.
2. Ensure `main` branch exists.
3. Go to **GitHub → Actions** and verify workflows run.
4. After `docker-ghcr` succeeds, your image is available on GHCR.

### Run deployed image from GHCR

```bash
docker pull ghcr.io/<owner>/<repo>:main
docker run --rm -p 8000:8000 ghcr.io/<owner>/<repo>:main
```

Now your app runs from the image built automatically from GitHub.
