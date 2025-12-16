import sys
import pickle
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import model_loader  # noqa: E402
from app.model_loader import load_model_with_fallback  # noqa: E402


class DummyModel:
    def __init__(self, name: str):
        self.name = name


def _write_model(path: Path, name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(DummyModel(name), f)


def test_load_model_prefers_production(tmp_path, monkeypatch):
    prod_dir = tmp_path / "models" / "production"
    fallback_dir = tmp_path / "models"
    prod_model = prod_dir / "regressor.pkl"
    fallback_model = fallback_dir / "regressor_random_forest.pkl"

    _write_model(prod_model, "prod")
    _write_model(fallback_model, "fallback")

    monkeypatch.setattr(model_loader, "PROD_DIR", prod_dir)
    monkeypatch.setattr(model_loader, "MODELS_DIR", fallback_dir)

    loaded = load_model_with_fallback(fallback_model, "regressor.pkl")
    assert isinstance(loaded, DummyModel)
    assert loaded.name == "prod"


def test_load_model_falls_back_when_no_production(tmp_path, monkeypatch):
    prod_dir = tmp_path / "models" / "production"
    fallback_dir = tmp_path / "models"
    fallback_model = fallback_dir / "regressor_random_forest.pkl"

    _write_model(fallback_model, "baseline")

    monkeypatch.setattr(model_loader, "PROD_DIR", prod_dir)
    monkeypatch.setattr(model_loader, "MODELS_DIR", fallback_dir)

    loaded = load_model_with_fallback(fallback_model, "regressor.pkl")
    assert isinstance(loaded, DummyModel)
    assert loaded.name == "baseline"


def test_load_model_raises_when_missing(tmp_path, monkeypatch):
    prod_dir = tmp_path / "models" / "production"
    fallback_dir = tmp_path / "models"
    fallback_model = fallback_dir / "missing.pkl"

    monkeypatch.setattr(model_loader, "PROD_DIR", prod_dir)
    monkeypatch.setattr(model_loader, "MODELS_DIR", fallback_dir)

    with pytest.raises(FileNotFoundError):
        load_model_with_fallback(fallback_model, "regressor.pkl")
