"""Model loading helpers with production fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib

from app.logging_config import configure_logging

logger = configure_logging().getChild("model_loader")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
PROD_DIR = MODELS_DIR / "production"


def load_model_with_fallback(fallback_path: Path, prod_name: str) -> Any:
    """
    Load a model from production if available, otherwise fall back to baseline.

    Parameters
    ----------
    fallback_path: Path
        Path to the baseline model (non-production).
    prod_name: str
        Filename inside models/production/ to attempt first.

    Returns
    -------
    Any
        The loaded model object.
    """
    prod_path = PROD_DIR / prod_name

    if prod_path.exists():
        logger.info("Loaded production model from %s", prod_path)
        return joblib.load(prod_path)

    if fallback_path.exists():
        logger.warning(
            "Production model not found at %s; falling back to baseline %s",
            prod_path,
            fallback_path,
        )
        return joblib.load(fallback_path)

    raise FileNotFoundError(
        f"No model found. Tried production {prod_path} and baseline {fallback_path}"
    )
