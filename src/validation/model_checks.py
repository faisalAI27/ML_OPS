"""Sanity checks for model predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def check_model_predictions(regressor, sample_features: pd.DataFrame, low: float = 1.0, high: float = 5.0) -> None:
    """
    Ensure model predictions fall within expected AQI bounds.

    Parameters
    ----------
    regressor : fitted model with predict
    sample_features : pd.DataFrame
        Feature rows to test predictions on.
    low, high : float
        Expected inclusive bounds for AQI predictions.
    """
    preds = regressor.predict(sample_features)
    preds = np.asarray(preds)
    if np.isnan(preds).any():
        raise ValueError("Model predictions contain NaN values")
    if not ((preds >= low) & (preds <= high)).all():
        raise ValueError(f"Model predictions outside expected range [{low}, {high}]")
