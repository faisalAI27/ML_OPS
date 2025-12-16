#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "== SmogGuard PK bootstrap =="

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR"
  $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

echo "Validating environment variables..."
if [ -z "${OPENWEATHER_API_KEY:-}" ]; then
  echo "WARNING: OPENWEATHER_API_KEY is not set. API calls will fail. Set it in .env or export it."
fi

echo "Checking model artifacts..."
if [ ! -f "models/production/regressor.pkl" ] || [ ! -f "models/production/classifier.pkl" ]; then
  echo "WARNING: Production models not found under models/production/. Using baseline fallback if present."
fi

echo "Running smoke test..."
python -m scripts.smoke_test

echo "Bootstrap completed."
