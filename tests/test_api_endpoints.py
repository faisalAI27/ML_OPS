import sys
import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import main  # noqa: E402


class DummyRegressor:
    def predict(self, X):
        return [4.2 for _ in range(len(X))]


import numpy as np


class DummyClassifier:
    def predict(self, X):
        return [1 for _ in range(len(X))]

    def predict_proba(self, X):
        return np.array([[0.1, 0.9] for _ in range(len(X))])


@pytest.fixture(autouse=True)
def patch_models(monkeypatch):
    monkeypatch.setattr(main, "reg_pipeline", DummyRegressor())
    monkeypatch.setattr(main, "clf_pipeline", DummyClassifier())
    yield


@pytest.fixture
def client():
    main._reset_metrics()
    return TestClient(main.app)


def test_health_root(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "ok"
    assert "models_loaded" in body


def test_model_info_with_no_file(monkeypatch, client, tmp_path):
    # Point PROD_DIR to temp without metadata
    monkeypatch.setattr(main, "PROD_DIR", tmp_path)
    resp = client.get("/model_info")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body


def test_model_info_with_file(monkeypatch, client, tmp_path):
    meta_dir = tmp_path
    meta_path = meta_dir / "model_metadata.json"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"model_version": "test", "note": "dummy"}))
    monkeypatch.setattr(main, "PROD_DIR", meta_dir)

    resp = client.get("/model_info")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("model_version") == "test"


def test_predict_realtime_success(monkeypatch, client):
    now_iso = datetime.utcnow().isoformat()

    def fake_poll(city):
        return {
            "aqi": 4,
            "components": {
                "co": 1.0,
                "no": 0.1,
                "no2": 0.2,
                "o3": 0.3,
                "so2": 0.4,
                "pm2_5": 10.0,
                "pm10": 20.0,
                "nh3": 0.5,
            },
            "observed_at_local": now_iso,
        }

    def fake_weather(city):
        return {
            "datetime_iso": now_iso,
            "temperature_2m": 25.0,
            "relative_humidity_2m": 50.0,
            "dew_point_2m": 10.0,
            "precipitation": 0.0,
            "surface_pressure": 1000.0,
            "wind_speed_10m": 2.0,
            "wind_direction_10m": 180.0,
            "shortwave_radiation": 100.0,
        }

    monkeypatch.setattr(main, "fetch_openweather_pollutants", fake_poll)
    monkeypatch.setattr(main, "fetch_openmeteo_weather", fake_weather)

    resp = client.get("/predict_realtime/Islamabad")
    assert resp.status_code == 200
    body = resp.json()
    assert "prediction" in body
    assert "realtime" in body
    assert body["prediction"]["hazard_label"] in ("Hazardous", "Safe / Moderate")


def test_predict_realtime_unsupported_city(client):
    resp = client.get("/predict_realtime/UnknownCity")
    assert resp.status_code == 400
    body = resp.json()
    assert "detail" in body


def test_metrics_lite(client, monkeypatch):
    now_iso = datetime.utcnow().isoformat()

    def fake_poll(city):
        return {
            "aqi": 4,
            "components": {"co": 1.0, "no": 0.1, "no2": 0.2, "o3": 0.3, "so2": 0.4, "pm2_5": 10.0, "pm10": 20.0, "nh3": 0.5},
            "observed_at_local": now_iso,
        }

    def fake_weather(city):
        return {
            "datetime_iso": now_iso,
            "temperature_2m": 25.0,
            "relative_humidity_2m": 50.0,
            "dew_point_2m": 10.0,
            "precipitation": 0.0,
            "surface_pressure": 1000.0,
            "wind_speed_10m": 2.0,
            "wind_direction_10m": 180.0,
            "shortwave_radiation": 100.0,
        }

    monkeypatch.setattr(main, "fetch_openweather_pollutants", fake_poll)
    monkeypatch.setattr(main, "fetch_openmeteo_weather", fake_weather)

    client.get("/predict_realtime/Islamabad")
    client.get("/predict_realtime/Lahore")

    # second call to ensure counters increment
    client.get("/predict_realtime/Islamabad")

    resp = client.get("/metrics-lite")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_requests"] >= 3
    assert "islamabad" in body["by_city"]
