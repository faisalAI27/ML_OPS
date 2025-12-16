from pathlib import Path
import numpy as np
import pandas as pd

from app.main import FEATURE_COLUMNS
from app.inference import reg_pipeline, clf_pipeline
from app.recommendations import build_recommendations


def sample_features_row() -> pd.DataFrame:
    data = {
        "main_aqi": 3.0,
        "components_co": 0.5,
        "components_no": 0.05,
        "components_no2": 0.1,
        "components_o3": 0.2,
        "components_so2": 0.05,
        "components_pm2_5": 15.0,
        "components_pm10": 30.0,
        "components_nh3": 0.05,
        "temperature_2m": 25.0,
        "relative_humidity_2m": 50.0,
        "dew_point_2m": 10.0,
        "precipitation": 0.0,
        "surface_pressure": 1000.0,
        "wind_speed_10m": 2.0,
        "wind_direction_10m": 180.0,
        "shortwave_radiation": 100.0,
        "city": "Islamabad",
        "hour": 12,
        "dayofweek": 1,
        "month": 6,
        "is_weekend": 0,
        "main_aqi_lag_1": 3.0,
        "main_aqi_lag_2": 3.0,
        "main_aqi_lag_3": 3.0,
        "components_pm2_5_lag_1": 15.0,
        "components_pm2_5_lag_2": 15.0,
        "components_pm2_5_lag_3": 15.0,
        "components_pm10_lag_1": 30.0,
        "components_pm10_lag_2": 30.0,
        "components_pm10_lag_3": 30.0,
        "components_pm2_5_roll3_mean": 15.0,
        "components_pm10_roll3_mean": 30.0,
    }
    return pd.DataFrame([data], columns=FEATURE_COLUMNS)


def run_model_behavior_checks():
    df = sample_features_row()
    aqi_pred = reg_pipeline.predict(df)[0]
    assert 1.0 <= aqi_pred <= 5.0, "AQI prediction out of range"

    if hasattr(clf_pipeline, "predict_proba"):
        proba = clf_pipeline.predict_proba(df)
        if hasattr(proba, "shape") and proba.shape[1] > 1:
            hazard_p = proba[0][1]
            assert 0.0 <= hazard_p <= 1.0, "Hazard prob out of [0,1]"

    recs = build_recommendations(aqi_3h=aqi_pred, hazard_label="Hazardous" if aqi_pred >= 4 else "Safe", features=df)
    assert recs.get("headline"), "Recommendations missing headline"
