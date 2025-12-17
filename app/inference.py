"""Inference utilities: load trained pipelines once at startup."""

from pathlib import Path

from app.logging_config import configure_logging
from app.model_loader import MODELS_DIR, load_model_with_fallback

logger = configure_logging().getChild("inference")

REG_MODEL_PATH = MODELS_DIR / "regressor_random_forest.pkl"
CLF_MODEL_PATH = MODELS_DIR / "classifier_random_forest.pkl"

class _DummyModel:
    """Lightweight placeholder used when model files are unavailable."""

    def __init__(self, name: str):
        self.name = name

    def predict(self, *_args, **_kwargs):  # pragma: no cover - defensive
        raise FileNotFoundError(f"{self.name} model is not available; ensure models are present before running inference.")


def _load_or_dummy(path: Path, prod_name: str, label: str):
    logger.info("Loading %s from production if available, fallback=%s", label, path)
    try:
        return load_model_with_fallback(path, prod_name)
    except FileNotFoundError:
        logger.warning("%s model missing; using dummy placeholder (tests will patch this)", label, exc_info=True)
        return _DummyModel(label)


# Load once on import with production fallback, otherwise use dummy to keep tests importable.
reg_pipeline = _load_or_dummy(REG_MODEL_PATH, "regressor.pkl", "regressor")
clf_pipeline = _load_or_dummy(CLF_MODEL_PATH, "classifier.pkl", "classifier")
