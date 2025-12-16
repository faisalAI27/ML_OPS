# SmogGuard PK – AQI Forecasting for 5 Pakistani Cities

Predicts 3-hour-ahead air quality (AQI 1–5 scale) and hazard status (AQI ≥ 4 → hazardous) for Islamabad, Lahore, Karachi, Peshawar, and Quetta. Includes data prep, training pipeline, model registry (files), FastAPI backend, Streamlit UI, CI/CD, and Docker.

## Project Overview
- Regression target: `main_aqi_t_plus_h` on a 1–5 scale (3 hours ahead).
- Classification: hazard = 1 if AQI ≥ 4 else 0.
- External data: OpenWeather (AQI + pollutants) and Open-Meteo (weather).

## Architecture Overview
```
Historical AQI + Weather --> Prefect Flow (aqi_training_flow)
                             - validation
                             - feature building
                             - model training
                             - evaluation & promotion
                                     |
                            models/production/*.pkl
                                     |
                           FastAPI backend (app/main.py)
                                     |
                            Streamlit UI (ui_app.py)
```
- Data → training → model registry (files under `models/production/`) → API → UI.
- Model loader prefers production models, falls back to baseline artifacts.

## Environment & Requirements
- Python 3.11 (local, Docker, CI)
- Install deps: `pip install -r requirements.txt`
- Environment variables (see `.env.example`):
  - `OPENWEATHER_API_KEY` (required)
  - `API_BASE_URL` (for UI when needed)
  - `NOTIFY_WEBHOOK_URL` (optional)
- Copy `.env.example` → `.env` and fill in your key. Docker Compose reads `OPENWEATHER_API_KEY` from the shell environment.
- If using Git LFS for models: `git lfs install` then `git lfs pull` to fetch production models.
- Training writes `data/reference_stats.json` for CI drift checks.

## How to Run (Local, no Docker)
```bash
# 1) Clone and enter
git clone <repo_url>
cd smogguard_pk

# 2) Create venv & install
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3) Set env
export OPENWEATHER_API_KEY=your_key_here   # or use .env + python-dotenv

# 4) (Optional) Train & promote models
python -m src.pipelines.aqi_flow

# 5) Run API
uvicorn app.main:app --reload

# 6) Run UI (new terminal)
source .venv/bin/activate
streamlit run ui_app.py
```
- API: http://127.0.0.1:8000
- UI: default Streamlit URL (shows predictions/recommendations)

## How to Run with Docker
```bash
OPENWEATHER_API_KEY=your_key_here docker compose up --build
```
- API: http://localhost:8000
- UI: http://localhost:8501
- Images: `smogguard_pk-api`, `smogguard_pk-ui`

## Endpoints
- `GET /health` – status, models_loaded, model_version.
- `GET /metrics-lite` – in-memory counters (total/success/fail/by_city).
- `GET /model_info` – metadata from `models/production/model_metadata.json` (metrics, training_run_id, etc.).
- `GET /predict_realtime/{city}` – city ∈ {Islamabad, Lahore, Karachi, Peshawar, Quetta}. Returns prediction (aqi_3h, hazard_prob, hazard_label) + realtime block + meta. Hazard rule: AQI ≥ 4 → “Hazardous”.

## Training & Scheduled Retraining
- Manual: `python -m src.pipelines.aqi_flow`
- CI: `.github/workflows/ci.yml` runs pytest on push/PR.
- Scheduled training: `.github/workflows/scheduled_training.yml` runs daily at 02:00 UTC (and via manual dispatch) to refresh `models/production/`.

## Monitoring & Logging
- Structured logging via `app/logging_config.py` (stdout, consistent format).
- Health/metrics: `/health`, `/metrics-lite` (counters reset on restart).
- Model loader logging shows which paths were used (production vs fallback).

## Project Structure (key parts)
```
app/
  main.py             # FastAPI endpoints
  inference.py        # Prediction logic
  external_clients.py # OpenWeather/Open-Meteo calls
  model_loader.py     # Production/baseline model loading
  logging_config.py   # Logging setup
src/
  pipelines/aqi_flow.py# Prefect training flow (validate, train, evaluate, promote)
  validation/         # Data/model validation checks
ui_app.py             # Streamlit UI
models/production/    # Deployed models + metadata
tests/                # Pytest suite
Dockerfile.api, Dockerfile.ui, docker-compose.yml
.github/workflows/    # CI + scheduled training
```

## Environment Variables
- Copy `.env.example` to `.env`.
- Set `OPENWEATHER_API_KEY`.
- `API_BASE_URL` can override UI backend URL (defaults to localhost).
- Docker Compose: set `OPENWEATHER_API_KEY` in your shell before `docker compose up`.

## Developer Shortcuts (Makefile)
- `make train` → `python -m src.pipelines.aqi_flow`
- `make api`   → `uvicorn app.main:app --reload`
- `make ui`    → `streamlit run ui_app.py`
- `make test`  → `pytest`

## Extending the Project
- New models: adjust `src/pipelines/aqi_flow.py`.
- New cities: update city config and ensure data availability.
- External monitoring (Prometheus, etc.) can be added later; current setup uses files for the model registry and live APIs for data.
