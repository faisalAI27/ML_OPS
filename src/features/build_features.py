"""Feature engineering for horizon-based forecasting."""

from __future__ import annotations

import pandas as pd


TIME_FEATURES = ["hour", "dayofweek", "month", "is_weekend"]
LAG_FEATURES = {
    "main_aqi": [1, 2, 3],
    "components_pm2_5": [1, 2, 3],
    "components_pm10": [1, 2, 3],
}
ROLLING_FEATURES = {
    "components_pm2_5": 3,
    "components_pm10": 3,
}


def _add_time_features(df: pd.DataFrame) -> None:
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)


def _add_lag_features(df: pd.DataFrame) -> list[str]:
    created: list[str] = []
    for col, lags in LAG_FEATURES.items():
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            df[lag_col] = df.groupby("city")[col].shift(lag)
            created.append(lag_col)
    return created


def _add_rolling_features(df: pd.DataFrame) -> list[str]:
    created: list[str] = []
    for col, window in ROLLING_FEATURES.items():
        if col not in df.columns:
            continue
        roll_col = f"{col}_roll{window}_mean"
        df[roll_col] = (
            df.groupby("city")[col]
            .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
        )
        created.append(roll_col)
    return created


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, and time-based features.

    - Requires datetime, city, main_aqi_t_plus_h, hazard_t_plus_h already present.
    - Keeps datetime for reference/splitting but not as a model feature.
    """
    required = {"datetime", "city", "main_aqi_t_plus_h", "hazard_t_plus_h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df.sort_values(["city", "datetime"])

    _add_time_features(df)
    lag_cols = _add_lag_features(df)
    roll_cols = _add_rolling_features(df)

    # Drop rows where engineered lags/rolling stats are unavailable
    engineered_cols = lag_cols + roll_cols
    if engineered_cols:
        df = df.dropna(subset=engineered_cols)

    return df
