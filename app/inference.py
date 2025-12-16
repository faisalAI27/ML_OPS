"""Inference utilities: load trained pipelines once at startup."""

from pathlib import Path

from app.logging_config import configure_logging
from app.model_loader import MODELS_DIR, load_model_with_fallback

logger = configure_logging().getChild("inference")

REG_MODEL_PATH = MODELS_DIR / "regressor_random_forest.pkl"
CLF_MODEL_PATH = MODELS_DIR / "classifier_random_forest.pkl"

# Load once on import with production fallback
logger.info("Loading regressor from production if available, fallback=%s", REG_MODEL_PATH)
reg_pipeline = load_model_with_fallback(REG_MODEL_PATH, "regressor.pkl")
logger.info("Loading classifier from production if available, fallback=%s", CLF_MODEL_PATH)
clf_pipeline = load_model_with_fallback(CLF_MODEL_PATH, "classifier.pkl")
