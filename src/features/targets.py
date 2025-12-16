"""Target generation utilities."""

from __future__ import annotations

import pandas as pd


def add_future_targets(df: pd.DataFrame, horizon_hours: int = 3) -> pd.DataFrame:
    """Add future main_aqi targets with a hazard indicator."""
    if "city" not in df.columns or "datetime" not in df.columns or "main_aqi" not in df.columns:
        raise ValueError("DataFrame must include city, datetime, and main_aqi columns")

    df = df.copy()
    df = df.sort_values(["city", "datetime"])

    df["main_aqi_t_plus_h"] = df.groupby("city")["main_aqi"].shift(-horizon_hours)
    df = df.dropna(subset=["main_aqi_t_plus_h"])

    df["hazard_t_plus_h"] = (df["main_aqi_t_plus_h"] >= 4).astype(int)
    return df
