import pandas as pd
import pytest

from src.validation.data_checks import validate_features_targets, validate_raw_data


def test_validate_raw_data_pass():
    df = pd.DataFrame(
        {
            "city": ["islamabad", "lahore"],
            "datetime": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "main_aqi": [3.0, 4.5],
            "components_pm2_5": [15.0, 20.0],
            "components_pm10": [30.0, 40.0],
        }
    )
    validate_raw_data(df)  # should not raise


def test_validate_raw_data_missing_cols():
    df = pd.DataFrame({"city": ["islamabad"], "main_aqi": [3.0]})
    with pytest.raises(ValueError):
        validate_raw_data(df)


def test_validate_raw_data_out_of_range():
    df = pd.DataFrame(
        {
            "city": ["islamabad"],
            "datetime": pd.to_datetime(["2024-01-01"]),
            "main_aqi": [6.0],
            "components_pm2_5": [15.0],
            "components_pm10": [30.0],
        }
    )
    with pytest.raises(ValueError):
        validate_raw_data(df)


def test_validate_features_targets_pass():
    X = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
    y_reg = pd.Series([1.5, 4.5])
    y_clf = pd.Series([0, 1])
    validate_features_targets(X, y_reg, y_clf)  # should not raise


def test_validate_features_targets_bad_labels():
    X = pd.DataFrame({"f1": [1], "f2": [3]})
    y_reg = pd.Series([2.0])
    y_clf = pd.Series([2])
    with pytest.raises(ValueError):
        validate_features_targets(X, y_reg, y_clf)
