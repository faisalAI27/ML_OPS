from pathlib import Path
import pandas as pd


def check_required_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def check_no_nulls(df: pd.DataFrame, cols: set[str]) -> None:
    if df[list(cols)].isna().any().any():
        raise ValueError("Null values found in required columns")


def check_ranges(df: pd.DataFrame) -> None:
    if not ((df["main_aqi"] >= 1) & (df["main_aqi"] <= 5)).all():
        raise ValueError("main_aqi out of [1,5] range")
    if "hazard_t_plus_h" in df.columns:
        if not set(df["hazard_t_plus_h"].unique()).issubset({0, 1}):
            raise ValueError("hazard_t_plus_h must be 0/1")


def run_data_integrity_checks(df: pd.DataFrame) -> None:
    required = {"city", "datetime", "main_aqi", "components_pm2_5", "components_pm10"}
    check_required_columns(df, required)
    check_no_nulls(df, required)
    check_ranges(df)
