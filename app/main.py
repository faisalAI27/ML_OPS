"""FastAPI service exposing +3h AQI and hazard predictions."""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import os
from collections import defaultdict

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi import Body

from app.logging_config import configure_logging
from app.model_loader import MODELS_DIR, PROD_DIR
from app.config import CITY_COORDS
from app.external_clients import fetch_openmeteo_weather, fetch_openweather_pollutants
from app.inference import clf_pipeline, reg_pipeline
from app.recommendations import build_recommendations
from app.schemas import (
    AQIPredictionRequest,
    AQIPredictionResponse,
    FeaturesPredictRequest,
    FeaturesPredictResponse,
    FeaturesBatchResponse,
)

logger = configure_logging().getChild("main")

FEATURE_COLUMNS = [
    "main_aqi",
    "components_co",
    "components_no",
    "components_no2",
    "components_o3",
    "components_so2",
    "components_pm2_5",
    "components_pm10",
    "components_nh3",
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "city",
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "main_aqi_lag_1",
    "main_aqi_lag_2",
    "main_aqi_lag_3",
    "components_pm2_5_lag_1",
    "components_pm2_5_lag_2",
    "components_pm2_5_lag_3",
    "components_pm10_lag_1",
    "components_pm10_lag_2",
    "components_pm10_lag_3",
    "components_pm2_5_roll3_mean",
    "components_pm10_roll3_mean",
]

app = FastAPI(
    title="SmogGuard PK – AQI Prediction API",
    version="0.1.0",
)

# In-memory request counters for metrics-lite
_total_requests = 0
_successful_requests = 0
_failed_requests = 0
_requests_by_city = defaultdict(int)


def _reset_metrics():
    global _total_requests, _successful_requests, _failed_requests, _requests_by_city
    _total_requests = 0
    _successful_requests = 0
    _failed_requests = 0
    _requests_by_city = defaultdict(int)


@app.get("/health")
def health():
    meta_path = PROD_DIR / "model_metadata.json"
    model_version = "unknown"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_version = meta.get("model_version") or meta.get("source_models", {}).get("model_version", "unknown")
        except Exception:
            model_version = "unknown"
    return {
        "status": "ok",
        "models_loaded": True,
        "model_version": model_version,
    }


@app.get("/meta")
def meta():
    return {
        "model_version": "0.1.0",
        "horizon_hours": 3,
        "test_regression": {
            "rmse": 0.46460589139258535,
            "mae": 0.3053506111570628,
            "r2": 0.8168652044889098,
        },
        "test_classification": {
            "accuracy": 0.9042826946052752,
            "f1": 0.9288082299463413,
            "roc_auc": 0.9678310609190398,
        },
    }


@app.post("/predict", response_model=AQIPredictionResponse)
def predict_aqi(payload: AQIPredictionRequest):
    data = payload.model_dump()
    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

    y_pred_reg = float(reg_pipeline.predict(df)[0])
    y_pred_clf = int(clf_pipeline.predict(df)[0])

    hazard_proba = None
    if hasattr(clf_pipeline, "predict_proba"):
        proba = clf_pipeline.predict_proba(df)
        if proba.shape[1] > 1:
            hazard_proba = float(proba[0][1])

    hazard_label = "hazardous" if y_pred_clf == 1 else "safe/moderate"

    return AQIPredictionResponse(
        city=payload.city,
        predicted_main_aqi_t_plus_h=y_pred_reg,
        predicted_hazard_t_plus_h=y_pred_clf,
        hazard_label=hazard_label,
        hazard_probability=hazard_proba,
        regression_rmse_test=0.46460589139258535,
        classification_accuracy_test=0.9042826946052752,
    )


@app.get("/predict_realtime/{city}")
def predict_realtime(city: str):
    global _total_requests, _successful_requests, _failed_requests, _requests_by_city
    city_norm = city.strip().lower()
    if city_norm not in CITY_COORDS:
        raise HTTPException(status_code=400, detail="Unsupported city")

    logger.info(
        "predict_realtime request city=%s, openweather_key=%s, prod_reg_exists=%s, prod_clf_exists=%s",
        city_norm,
        bool(os.getenv("OPENWEATHER_API_KEY")),
        (PROD_DIR / "regressor.pkl").exists(),
        (PROD_DIR / "classifier.pkl").exists(),
    )

    try:
        poll = fetch_openweather_pollutants(city_norm)
        weather = fetch_openmeteo_weather(city_norm)
    except Exception as exc:  # pragma: no cover - passthrough to HTTP
        logger.exception("Failed to fetch real-time data for %s", city_norm)
        _total_requests += 1
        _failed_requests += 1
        _requests_by_city[city_norm] += 1
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch real-time data from upstream APIs",
        ) from exc

    comps = poll["components"]
    main_aqi = float(poll["aqi"])

    dt = datetime.fromisoformat(weather["datetime_iso"])
    hour = dt.hour
    dayofweek = dt.weekday()
    month = dt.month
    is_weekend = 1 if dayofweek >= 5 else 0

    observed_local = datetime.fromisoformat(poll["observed_at_local"])
    predicted_for_local = observed_local + timedelta(hours=3)

    pm25 = comps.get("pm2_5", 0.0)
    pm10 = comps.get("pm10", 0.0)

    city_for_model = city_norm.title()

    row = {
        "main_aqi": main_aqi,
        "components_co": comps.get("co", 0.0),
        "components_no": comps.get("no", 0.0),
        "components_no2": comps.get("no2", 0.0),
        "components_o3": comps.get("o3", 0.0),
        "components_so2": comps.get("so2", 0.0),
        "components_pm2_5": pm25,
        "components_pm10": pm10,
        "components_nh3": comps.get("nh3", 0.0),
        **{k: v for k, v in weather.items() if k != "datetime_iso"},
        "city": city_for_model,
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "is_weekend": is_weekend,
        "main_aqi_lag_1": main_aqi,
        "main_aqi_lag_2": main_aqi,
        "main_aqi_lag_3": main_aqi,
        "components_pm2_5_lag_1": pm25,
        "components_pm2_5_lag_2": pm25,
        "components_pm2_5_lag_3": pm25,
        "components_pm10_lag_1": pm10,
        "components_pm10_lag_2": pm10,
        "components_pm10_lag_3": pm10,
        "components_pm2_5_roll3_mean": pm25,
        "components_pm10_roll3_mean": pm10,
    }

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    try:
        aqi_3h = float(reg_pipeline.predict(df)[0])

        hazard_proba = None
        if hasattr(clf_pipeline, "predict_proba"):
            proba = clf_pipeline.predict_proba(df)
            if hasattr(proba, "shape") and proba.shape[1] > 1:
                hazard_proba = float(proba[0][1])
        hazard_prob = float(hazard_proba) if hazard_proba is not None else None
    except FileNotFoundError as exc:
        logger.exception("Model files missing during inference")
        _total_requests += 1
        _failed_requests += 1
        _requests_by_city[city_norm] += 1
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to run inference for %s", city_norm)
        _total_requests += 1
        _failed_requests += 1
        _requests_by_city[city_norm] += 1
        raise HTTPException(status_code=500, detail="Failed to run inference") from exc

    hazard_label = "Hazardous" if aqi_3h >= 4.0 else "Safe / Moderate"

    recs = build_recommendations(
        aqi_3h=aqi_3h,
        hazard_label=hazard_label,
        current_aqi=main_aqi,
        horizon_hours=3,
        features=df,
    )

    _total_requests += 1
    _successful_requests += 1
    _requests_by_city[city_norm] += 1

    meta_data = {
        "model_version": "0.1.0",
        "horizon_hours": 3,
        "test_regression": {
            "rmse": 0.46460589139258535,
            "mae": 0.3053506111570628,
            "r2": 0.8168652044889098,
        },
        "test_classification": {
            "accuracy": 0.9042826946052752,
            "f1": 0.9288082299463413,
            "roc_auc": 0.9678310609190398,
        },
    }

    return {
        "city": city_for_model,
        "prediction": {
            "aqi_3h": aqi_3h,
            "hazard_prob": hazard_prob,
            "hazard_label": hazard_label,
        },
        "realtime": {
            "current_aqi": poll["aqi"],
            "observed_at_local": observed_local.isoformat(),
            "predicted_for_local": predicted_for_local.isoformat(),
            "scale": "OpenWeather 1–5 AQI",
            "scale_description": "1=Good, 2=Fair, 3=Moderate, 4=Poor, 5=Very Poor",
        },
        "meta": meta_data,
        "recommendations": recs,
    }


@app.get("/metrics-lite")
def metrics_lite():
    return {
        "total_requests": _total_requests,
        "successful_requests": _successful_requests,
        "failed_requests": _failed_requests,
        "by_city": dict(_requests_by_city),
    }


def _build_prediction_response(df_row: pd.DataFrame, city_norm: str, input_type: str):
    try:
        aqi_3h = float(reg_pipeline.predict(df_row)[0])

        hazard_proba = None
        if hasattr(clf_pipeline, "predict_proba"):
            proba = clf_pipeline.predict_proba(df_row)
            if hasattr(proba, "shape") and proba.shape[1] > 1:
                hazard_proba = float(proba[0][1])
        hazard_prob = float(hazard_proba) if hazard_proba is not None else None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Failed to run inference") from exc

    hazard_label = "Hazardous" if aqi_3h >= 4.0 else "Safe / Moderate"
    recs = build_recommendations(
        aqi_3h=aqi_3h,
        hazard_label=hazard_label,
        current_aqi=df_row.get("main_aqi", [None])[0] if hasattr(df_row, "get") else None,
        horizon_hours=3,
        features=df_row,
    )

    meta_data = {
        "model_version": "0.1.0",
        "horizon_hours": 3,
        "test_regression": {
            "rmse": 0.46460589139258535,
            "mae": 0.3053506111570628,
            "r2": 0.8168652044889098,
        },
        "test_classification": {
            "accuracy": 0.9042826946052752,
            "f1": 0.9288082299463413,
            "roc_auc": 0.9678310609190398,
        },
        "input_type": input_type,
    }

    return {
        "city": city_norm.title() if city_norm else "",
        "prediction": {
            "aqi_3h": aqi_3h,
            "hazard_prob": hazard_prob,
            "hazard_label": hazard_label,
        },
        "realtime": {},
        "meta": meta_data,
        "recommendations": recs,
    }


@app.post("/predict_from_features", response_model=FeaturesPredictResponse)
def predict_from_features(payload: FeaturesPredictRequest):
    data = payload.features
    missing = [c for c in FEATURE_COLUMNS if c not in data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required feature columns: {missing}")

    df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
    result = _build_prediction_response(df, data.get("city", ""), "features_json")
    return {"result": result, "input_type": "features_json"}


@app.post("/predict_from_csv", response_model=FeaturesBatchResponse)
async def predict_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required feature columns: {missing}")

    results = []
    for _, row in df.iterrows():
        row_df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
        res = _build_prediction_response(row_df, row.get("city", ""), "features_csv")
        results.append(res)

    return {"results": results, "input_type": "features_csv"}


@app.get("/model_info")
def model_info():
    """Return production model metadata if available."""
    meta_path = PROD_DIR / "model_metadata.json"
    if not meta_path.exists():
        return {
            "status": "no_production_metadata",
            "message": f"{meta_path} not found. Using baseline models.",
        }
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to read production metadata from %s", meta_path)
        return {
            "status": "error",
            "message": f"Could not parse {meta_path}: {exc}",
        }
