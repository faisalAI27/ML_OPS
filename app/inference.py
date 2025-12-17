"""Inference utilities: lazy-load trained pipelines with clear errors when missing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.logging_config import configure_logging
from app.model_loader import MODELS_DIR, load_model_with_fallback

logger = configure_logging().getChild("inference")

REG_MODEL_PATH = MODELS_DIR / "regressor_random_forest.pkl"
CLF_MODEL_PATH = MODELS_DIR / "classifier_random_forest.pkl"


class MissingModelError(FileNotFoundError):
    """Raised when no trained model artifacts are available."""


_reg_cache: Optional[Any] = None
_clf_cache: Optional[Any] = None


def _load_model(path: Path, prod_name: str, label: str):
    logger.info("Loading %s from production if available, fallback=%s", label, path)
    try:
        return load_model_with_fallback(path, prod_name)
    except FileNotFoundError as exc:  # pragma: no cover - runtime error path
        raise MissingModelError(
            f"{label} model is not available. Run `python -m src.pipelines.aqi_flow` to train models."
        ) from exc


def get_reg_pipeline():
    """Return the regressor pipeline, loading on first use."""
    global _reg_cache
    if _reg_cache is None:
        _reg_cache = _load_model(REG_MODEL_PATH, "regressor.pkl", "regressor")
    return _reg_cache


def get_clf_pipeline():
    """Return the classifier pipeline, loading on first use."""
    global _clf_cache
    if _clf_cache is None:
        _clf_cache = _load_model(CLF_MODEL_PATH, "classifier.pkl", "classifier")
    return _clf_cache


def _clear_model_cache():
    """Test helper to reset cached models."""
    global _reg_cache, _clf_cache
    _reg_cache = None
    _clf_cache = None
