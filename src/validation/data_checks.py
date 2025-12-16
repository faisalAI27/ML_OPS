"""Simple data validation utilities for AQI datasets."""

from __future__ import annotations

import pandas as pd


def validate_raw_data(df: pd.DataFrame) -> None:
    """Validate raw dataframe shape and critical columns."""
    required_cols = {"city", "datetime", "main_aqi", "components_pm2_5", "components_pm10"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {missing}")

    critical = df[list(required_cols)]
    if critical.isna().any().any():
        raise ValueError("Raw data contains nulls in critical columns")

    if not ((critical["main_aqi"] >= 1) & (critical["main_aqi"] <= 5)).all():
        raise ValueError("main_aqi values must be within [1, 5]")


def validate_features_targets(features_df: pd.DataFrame, y_reg: pd.Series, y_clf: pd.Series) -> None:
    """Validate feature and target integrity."""
    if features_df.isna().any().any():
        raise ValueError("Features contain NaN values")

    if not ((y_reg >= 1) & (y_reg <= 5)).all():
        raise ValueError("Regression target (AQI) must be within [1, 5]")

    unique_cls = set(y_clf.unique())
    if not unique_cls.issubset({0, 1}):
        raise ValueError(f"Hazard labels must be 0/1. Found {unique_cls}")
