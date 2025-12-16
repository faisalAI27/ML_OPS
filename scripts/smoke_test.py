"""Smoke test for SmogGuard PK API without starting a server."""

from app.main import app
from fastapi.testclient import TestClient
from app import main as app_main
from datetime import datetime
import pandas as pd


def run():
    client = TestClient(app)

    # Health
    r = client.get("/health")
    if r.status_code != 200:
        raise SystemExit(f"/health failed: {r.status_code} {r.text}")

    # Mock external calls for predict_realtime
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

    # Patch
    app_main.fetch_openweather_pollutants = fake_poll  # type: ignore
    app_main.fetch_openmeteo_weather = fake_weather  # type: ignore

    # Call prediction
    r = client.get("/predict_realtime/Islamabad")
    if r.status_code != 200:
        raise SystemExit(f"/predict_realtime failed: {r.status_code} {r.text}")

    print("Smoke test passed.")


if __name__ == "__main__":
    run()
