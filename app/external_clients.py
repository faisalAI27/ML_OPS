"""External API clients for pollutant and weather data."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import requests

from app.config import CITY_COORDS, OPENWEATHER_API_KEY
from app.logging_config import configure_logging

logger = configure_logging().getChild("external_clients")

PK_TZ = timezone(timedelta(hours=5))


def fetch_openweather_pollutants(city: str) -> Dict[str, Any]:
    """Fetch current pollutant readings for a normalized city key."""
    if not OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set")
    if city not in CITY_COORDS:
        raise ValueError(f"Unsupported city: {city}")

    coords = CITY_COORDS[city]
    lat, lon = coords["lat"], coords["lon"]

    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    logger.info("Calling OpenWeather for city=%s lat=%s lon=%s", city, lat, lon)
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    entry = data["list"][0]
    aqi = entry["main"]["aqi"]
    comp = entry["components"]
    ts_unix = entry.get("dt")
    observed_at_utc = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
    observed_at_pk = observed_at_utc.astimezone(PK_TZ)

    return {
        "aqi": aqi,
        "components": comp,
        "observed_at_utc": observed_at_utc.isoformat(),
        "observed_at_local": observed_at_pk.isoformat(),
    }


def fetch_openmeteo_weather(city: str) -> Dict[str, Any]:
    """Fetch current weather for a normalized city key."""
    coords = CITY_COORDS[city]
    lat, lon = coords["lat"], coords["lon"]

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "precipitation",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
        ],
        "timezone": "auto",
    }
    logger.info("Calling Open-Meteo for city=%s lat=%s lon=%s", city, lat, lon)
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    idx = -1

    return {
        "temperature_2m": hourly["temperature_2m"][idx],
        "relative_humidity_2m": hourly["relative_humidity_2m"][idx],
        "dew_point_2m": hourly["dew_point_2m"][idx],
        "precipitation": hourly["precipitation"][idx],
        "surface_pressure": hourly["surface_pressure"][idx],
        "wind_speed_10m": hourly["wind_speed_10m"][idx],
        "wind_direction_10m": hourly["wind_direction_10m"][idx],
        "shortwave_radiation": hourly["shortwave_radiation"][idx],
        "datetime_iso": hourly["time"][idx],
    }
