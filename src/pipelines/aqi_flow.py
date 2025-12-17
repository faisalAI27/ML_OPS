"""Prefect flow for AQI training using the existing baseline pipeline.

This orchestrates: load data -> feature/target prep -> train RF models ->
evaluate -> register production artifacts. It reuses logic from src.models.train_model.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from prefect import flow, task
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# Make sure the repo root is importable even if the flow is run outside the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR
from src.models.train_model import (
    CUTOFF_TS,
    HORIZON_HOURS,
    evaluate_classification,
    evaluate_regression,
    fit_baseline_models,
    load_and_prepare_data,
    make_preprocessor,
    prepare_train_val_split,
)
from src.notifications import send_webhook
from src.ml_tests.reference_builder import save_reference_stats, save_current_sample
from src.ml_tests.drift_checks import REFERENCE_PATH
from src.validation.data_checks import validate_features_targets, validate_raw_data
from src.validation.model_checks import check_model_predictions

METRICS_DIR = Path("reports/metrics")


def _aqi_to_bucket(aqi: float) -> int:
    """Map AQI 1-5 to severity bucket 0..3 (1->0, 2->1, 3->2, 4-5->3)."""
    try:
        val = int(round(float(aqi)))
    except Exception:
        val = 3
    val = max(1, min(5, val))
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 3}
    return mapping.get(val, 2)


def _get_run_id_safe() -> str | None:
    """Return a flow run id if available; avoid hard dependency on runtime context."""
    try:
        from prefect.runtime import flow_run

        return flow_run.get_id()
    except Exception:
        return None


@task(retries=3, retry_delay_seconds=30, timeout_seconds=600)
def load_raw_data():
    """Load the cleaned historical dataset using the baseline loader."""
    return load_and_prepare_data(horizon_hours=HORIZON_HOURS)


@task(retries=3, retry_delay_seconds=30, timeout_seconds=600)
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


@task(retries=3, retry_delay_seconds=30, timeout_seconds=300)
def validate_raw_task(df):
    """Validate raw dataframe before feature engineering."""
    validate_raw_data(df)
    return df


@task(retries=3, retry_delay_seconds=30, timeout_seconds=300)
def validate_features_task(X_train, y_reg_train, y_clf_train):
    """Validate feature matrix and targets."""
    validate_features_targets(X_train, y_reg_train, y_clf_train)
    return True


@task(retries=2, retry_delay_seconds=30, timeout_seconds=1800)
def train_models_task(X_train, y_reg_train, y_clf_train, feature_columns) -> Dict[str, Any]:
    """Train baseline RF models and a recommender classifier; persist to disk."""
    reg_model, clf_model = fit_baseline_models(X_train, y_reg_train, y_clf_train, feature_columns)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    reg_path = MODELS_DIR / "regressor_random_forest.pkl"
    clf_path = MODELS_DIR / "classifier_random_forest.pkl"
    joblib.dump(reg_model, reg_path)
    joblib.dump(clf_model, clf_path)

    # Recommender: bucket severity from AQI and train a simple classifier
    y_rec_train = y_reg_train.map(_aqi_to_bucket)
    rec_preprocessor = make_preprocessor(feature_columns)
    rec_model = Pipeline(
        steps=[
            ("preprocessor", rec_preprocessor),
            ("model", DecisionTreeClassifier(random_state=42)),
        ]
    )
    rec_model.fit(X_train, y_rec_train)
    rec_path = MODELS_DIR / "recommender.pkl"
    joblib.dump(rec_model, rec_path)

    return {
        "regressor_path": str(reg_path),
        "classifier_path": str(clf_path),
        "recommender_path": str(rec_path),
        "feature_columns": feature_columns,
    }


@task(retries=2, retry_delay_seconds=30, timeout_seconds=600)
def check_model_task(regressor_path: str, sample_features):
    """Run prediction sanity checks on the trained regressor."""
    reg_model = joblib.load(regressor_path)
    # Use a small sample to validate outputs
    sample = sample_features.head(50)
    check_model_predictions(reg_model, sample)
    return True


@task(retries=2, retry_delay_seconds=30, timeout_seconds=900)
def evaluate_models_task(models_info: Dict[str, Any], X_val, y_reg_val, y_clf_val) -> Dict[str, Any]:
    """Evaluate saved models on the validation split and persist metrics."""
    reg_model = joblib.load(models_info["regressor_path"])
    clf_model = joblib.load(models_info["classifier_path"])
    rec_model = joblib.load(models_info["recommender_path"])

    reg_pred = reg_model.predict(X_val)
    clf_pred = clf_model.predict(X_val)
    clf_proba = None
    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(X_val)
        if proba.shape[1] > 1:
            clf_proba = proba[:, 1]

    reg_metrics = evaluate_regression(y_reg_val, reg_pred)
    clf_metrics = evaluate_classification(y_clf_val, clf_pred, clf_proba)
    y_rec_val = y_reg_val.map(_aqi_to_bucket)
    rec_pred = rec_model.predict(X_val)
    rec_accuracy = float((rec_pred == y_rec_val).mean())

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat().replace(":", "-")
    metrics_path = METRICS_DIR / f"metrics_{ts}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "regression": reg_metrics,
                "classification": clf_metrics,
                "recommender": {"accuracy": rec_accuracy},
                "created_at": datetime.utcnow().isoformat() + "Z",
            },
            f,
            indent=2,
        )

    return {
        "regression": reg_metrics,
        "classification": clf_metrics,
        "recommender": {"accuracy": rec_accuracy},
        "path": str(metrics_path),
    }


@task(retries=2, retry_delay_seconds=30, timeout_seconds=600)
def register_production_model(models_info: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Copy trained models to production locations and save metadata."""
    run_id = _get_run_id_safe()

    prod_dir = MODELS_DIR / "production"
    prod_dir.mkdir(parents=True, exist_ok=True)

    reg_prod = prod_dir / "regressor.pkl"
    clf_prod = prod_dir / "classifier.pkl"
    rec_prod = prod_dir / "recommender.pkl"
    shutil.copy(models_info["regressor_path"], reg_prod)
    shutil.copy(models_info["classifier_path"], clf_prod)
    shutil.copy(models_info["recommender_path"], rec_prod)

    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source_models": models_info,
        "metrics": metrics,
        "training_run_id": run_id or datetime.utcnow().isoformat() + "Z",
        "recommender": {
            "path": str(rec_prod),
            "model_type": "DecisionTreeClassifier",
        },
    }
    with open(prod_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "regressor": str(reg_prod),
        "classifier": str(clf_prod),
        "recommender": str(rec_prod),
        "metadata": str(prod_dir / "model_metadata.json"),
    }


@flow(name="aqi_training_flow")
def aqi_training_flow():
    """Prefect flow orchestrating AQI training/evaluation."""
    try:
        df = load_raw_data()
        validate_raw_task(df)
        X_train, y_reg_train, y_clf_train, X_val, y_reg_val, y_clf_val, feature_cols = build_features_targets(df)
        validate_features_task(X_train, y_reg_train, y_clf_train)
        # Save reference stats for drift checks
        save_reference_stats(pd.concat([X_train, X_val]), feature_cols, REFERENCE_PATH)
        # Save deterministic current sample for CI drift checks
        save_current_sample(pd.concat([X_train, X_val]), Path("tests/data/current_sample.csv"))
        models_info = train_models_task(X_train, y_reg_train, y_clf_train, feature_cols)
        check_model_task(models_info["regressor_path"], X_val)
        metrics = evaluate_models_task(models_info, X_val, y_reg_val, y_clf_val)
        prod_info = register_production_model(models_info, metrics)

        summary = (
            f"Regression metrics: {metrics.get('regression')}, "
            f"Classification metrics: {metrics.get('classification')}, "
            f"Production artifacts: {prod_info}"
        )
        print("\n=== AQI Training Flow Summary ===")
        print(summary)
        send_webhook(f"Training succeeded. {summary}", status="success")
    except Exception as exc:
        send_webhook(f"Training failed: {exc}", status="error")
        raise


if __name__ == "__main__":
    aqi_training_flow()
