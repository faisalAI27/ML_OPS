"""Prefect flow for AQI training using the existing baseline pipeline.

This orchestrates: load data -> feature/target prep -> train RF models ->
evaluate -> register production artifacts. It reuses logic from src.models.train_model.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
from prefect import flow, get_run_context, task

from src.config import MODELS_DIR
from src.models.train_model import (
    CUTOFF_TS,
    HORIZON_HOURS,
    evaluate_classification,
    evaluate_regression,
    fit_baseline_models,
    load_and_prepare_data,
    prepare_train_val_split,
)
from src.validation.data_checks import validate_features_targets, validate_raw_data
from src.validation.model_checks import check_model_predictions

METRICS_DIR = Path("reports/metrics")


@task
def load_raw_data():
    """Load the cleaned historical dataset using the baseline loader."""
    return load_and_prepare_data(horizon_hours=HORIZON_HOURS)


@task
def build_features_targets(df):
    """Create train/val splits and targets matching the baseline pipeline."""
    (
        X_train,
        y_reg_train,
        y_clf_train,
        X_val,
        y_reg_val,
        y_clf_val,
        feature_columns,
    ) = prepare_train_val_split(df, cutoff_ts=CUTOFF_TS)

    return X_train, y_reg_train, y_clf_train, X_val, y_reg_val, y_clf_val, feature_columns


@task
def validate_raw_task(df):
    """Validate raw dataframe before feature engineering."""
    validate_raw_data(df)
    return df


@task
def validate_features_task(X_train, y_reg_train, y_clf_train):
    """Validate feature matrix and targets."""
    validate_features_targets(X_train, y_reg_train, y_clf_train)
    return True


@task
def train_models_task(X_train, y_reg_train, y_clf_train, feature_columns) -> Dict[str, Any]:
    """Train baseline RF models and persist them to disk."""
    reg_model, clf_model = fit_baseline_models(X_train, y_reg_train, y_clf_train, feature_columns)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    reg_path = MODELS_DIR / "regressor_random_forest.pkl"
    clf_path = MODELS_DIR / "classifier_random_forest.pkl"
    joblib.dump(reg_model, reg_path)
    joblib.dump(clf_model, clf_path)

    return {
        "regressor_path": str(reg_path),
        "classifier_path": str(clf_path),
        "feature_columns": feature_columns,
    }


@task
def check_model_task(regressor_path: str, sample_features):
    """Run prediction sanity checks on the trained regressor."""
    reg_model = joblib.load(regressor_path)
    # Use a small sample to validate outputs
    sample = sample_features.head(50)
    check_model_predictions(reg_model, sample)
    return True


@task
def evaluate_models_task(models_info: Dict[str, Any], X_val, y_reg_val, y_clf_val) -> Dict[str, Any]:
    """Evaluate saved models on the validation split and persist metrics."""
    reg_model = joblib.load(models_info["regressor_path"])
    clf_model = joblib.load(models_info["classifier_path"])

    reg_pred = reg_model.predict(X_val)
    clf_pred = clf_model.predict(X_val)
    clf_proba = None
    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(X_val)
        if proba.shape[1] > 1:
            clf_proba = proba[:, 1]

    reg_metrics = evaluate_regression(y_reg_val, reg_pred)
    clf_metrics = evaluate_classification(y_clf_val, clf_pred, clf_proba)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat().replace(":", "-")
    metrics_path = METRICS_DIR / f"metrics_{ts}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "regression": reg_metrics,
                "classification": clf_metrics,
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
            f,
            indent=2,
        )

    return {"regression": reg_metrics, "classification": clf_metrics, "path": str(metrics_path)}


@task
def register_production_model(models_info: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Copy trained models to production locations and save metadata."""
    run_id = None
    try:
        ctx = get_run_context()
        run_id = ctx.flow_run.id if ctx else None
    except Exception:
        run_id = None

    prod_dir = MODELS_DIR / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    reg_prod = prod_dir / "regressor.pkl"
    clf_prod = prod_dir / "classifier.pkl"
    shutil.copy(models_info["regressor_path"], reg_prod)
    shutil.copy(models_info["classifier_path"], clf_prod)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_models": models_info,
        "metrics": metrics,
        "training_run_id": run_id or datetime.utcnow().isoformat() + "Z",
    }
    with open(prod_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"regressor": str(reg_prod), "classifier": str(clf_prod), "metadata": str(prod_dir / "model_metadata.json")}


@flow(name="aqi_training_flow")
def aqi_training_flow():
    """Prefect flow orchestrating AQI training/evaluation."""
    df = load_raw_data()
    validate_raw_task(df)
    X_train, y_reg_train, y_clf_train, X_val, y_reg_val, y_clf_val, feature_cols = build_features_targets(df)
    validate_features_task(X_train, y_reg_train, y_clf_train)
    models_info = train_models_task(X_train, y_reg_train, y_clf_train, feature_cols)
    check_model_task(models_info["regressor_path"], X_val)
    metrics = evaluate_models_task(models_info, X_val, y_reg_val, y_clf_val)
    prod_info = register_production_model(models_info, metrics)

    print("\n=== AQI Training Flow Summary ===")
    print(f"Regression metrics: {metrics.get('regression')}")
    print(f"Classification metrics: {metrics.get('classification')}")
    print(f"Production artifacts: {prod_info}")


if __name__ == "__main__":
    aqi_training_flow()
