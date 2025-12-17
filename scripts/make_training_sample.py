"""Create a lightweight training sample CSV for CI/scheduled runs.

Usage:
  python scripts/make_training_sample.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.train_model import TRAINING_FILE, TRAINING_SAMPLE_FILE


def generate_synthetic_sample(rows: int = 600, cities: list[str] | None = None) -> pd.DataFrame:
    """Build a deterministic synthetic dataset matching expected columns."""
    rng = np.random.default_rng(42)
    cities = cities or ["Islamabad", "Lahore", "Karachi"]

    records = []
    for city in cities:
        for i in range(rows // len(cities)):
            dt = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i)
            base_aqi = 2 + 0.02 * i
            records.append(
                {
                    "datetime": dt,
                    "city": city,
                    "main_aqi": float(np.clip(base_aqi + rng.normal(0, 0.5), 1, 5)),
                    "components_co": float(rng.uniform(0.1, 1.0)),
                    "components_no": float(rng.uniform(0.01, 0.2)),
                    "components_no2": float(rng.uniform(0.05, 0.3)),
                    "components_o3": float(rng.uniform(0.05, 0.4)),
                    "components_so2": float(rng.uniform(0.01, 0.2)),
                    "components_pm2_5": float(np.clip(10 + rng.normal(0, 3), 0, 80)),
                    "components_pm10": float(np.clip(20 + rng.normal(0, 5), 0, 120)),
                    "components_nh3": float(rng.uniform(0.01, 0.1)),
                    "temperature_2m": float(rng.uniform(5, 40)),
                    "relative_humidity_2m": float(rng.uniform(20, 90)),
                    "dew_point_2m": float(rng.uniform(-5, 25)),
                    "precipitation": float(np.clip(rng.normal(0.1, 0.05), 0, 5)),
                    "surface_pressure": float(rng.uniform(950, 1050)),
                    "wind_speed_10m": float(rng.uniform(0, 12)),
                    "wind_direction_10m": float(rng.uniform(0, 360)),
                    "shortwave_radiation": float(rng.uniform(0, 500)),
                }
            )
    return pd.DataFrame.from_records(records)


def make_sample(sample_size: int = 5000) -> Path:
    TRAINING_SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if TRAINING_FILE.exists():
        df = pd.read_parquet(TRAINING_FILE)
        sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    else:
        sample = generate_synthetic_sample(rows=min(sample_size, 6000))

    sample.to_csv(TRAINING_SAMPLE_FILE, index=False)
    return TRAINING_SAMPLE_FILE


def main():
    parser = argparse.ArgumentParser(description="Create training_sample.csv from full parquet (if available).")
    parser.add_argument("--rows", type=int, default=5000, help="Number of rows to include in the sample.")
    args = parser.parse_args()

    path = make_sample(sample_size=args.rows)
    print(f"Wrote sample to {path}")


if __name__ == "__main__":
    main()
