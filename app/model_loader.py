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


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(200)
        return b"git-lfs.github.com/spec" in head
    except Exception:
        return False


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
        if _is_lfs_pointer(prod_path):
            logger.warning("Production model at %s is an LFS pointer/invalid; falling back", prod_path)
        else:
            try:
                logger.info("Loaded production model from %s", prod_path)
                return joblib.load(prod_path)
            except Exception:
                logger.warning("Failed to load production model at %s; falling back", prod_path)

    if fallback_path.exists():
        logger.warning(
            "Production model not found at %s; falling back to baseline %s",
            prod_path,
            fallback_path,
        )
        if _is_lfs_pointer(fallback_path):
            logger.error("Baseline model at %s is an LFS pointer/invalid", fallback_path)
        else:
            try:
                return joblib.load(fallback_path)
            except Exception as exc:
                logger.error("Failed to load baseline model at %s: %s", fallback_path, exc)

    raise FileNotFoundError(
        f"No model found. Tried production {prod_path} and baseline {fallback_path}"
    )
