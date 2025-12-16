"""Central configuration for app-wide constants."""

import os
from dotenv import load_dotenv

# Load .env at startup so OPENWEATHER_API_KEY is available without manual export.
load_dotenv()

CITY_COORDS = {
    "islamabad": {"lat": 33.6844, "lon": 73.0479},
    "lahore": {"lat": 31.5204, "lon": 74.3587},
    "karachi": {"lat": 24.8607, "lon": 67.0011},
    "peshawar": {"lat": 34.0151, "lon": 71.5249},
    "quetta": {"lat": 30.1798, "lon": 66.9750},
}

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
