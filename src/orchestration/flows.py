"""Prefect flows for feature building, training, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.config import (
    PROCESSED_DIR,
    PRODUCTION_MODELS_DIR,
    RAW_TESTING_DIR,
    RAW_TRAINING_DIR,
)
from src.features.build_features import build_features
from src.features.targets import add_future_targets
from src.models.train_model import (
    CUTOFF_TS as DEFAULT_CUTOFF_TS,
    HORIZON_HOURS as DEFAULT_HORIZON,
    evaluate_classification,
    evaluate_regression,
    make_preprocessor,
)


TRAINING_FILE = RAW_TRAINING_DIR / "training_all_cities_until_2024_06_30.parquet"
TEST_FILE = RAW_TESTING_DIR / "testing_all_cities_2024_07_to_12.parquet"
PROCESSED_TRAINING = PROCESSED_DIR / "training_features.parquet"


@task
def load_raw_training_data(path: Path = TRAINING_FILE) -> pd.DataFrame:
    logger = get_run_logger()
    if not path.exists():
        raise FileNotFoundError(f"Training data not found at {path}")
    logger.info("Loading raw training data from %s", path)
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["datetime"])
    df["city"] = df["city"].astype(str)
    df = df.drop_duplicates(subset=["city", "datetime"]).sort_values(["city", "datetime"])
    return df


@task
def create_targets_task(df: pd.DataFrame, horizon_hours: int = DEFAULT_HORIZON) -> pd.DataFrame:
    return add_future_targets(df, horizon_hours=horizon_hours)


@task
def create_features_task(df: pd.DataFrame) -> pd.DataFrame:
    return build_features(df)


@task
def save_processed_dataset(df: pd.DataFrame, path: Path = PROCESSED_TRAINING) -> Path:
    logger = get_run_logger()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved processed training features to %s", path)
    return path


@flow
def build_offline_features_flow(horizon_hours: int = DEFAULT_HORIZON) -> str:
    """Build processed training features for offline model training."""
    df = load_raw_training_data()
    df = create_targets_task(df, horizon_hours=horizon_hours)
    df = create_features_task(df)
    output_path = save_processed_dataset(df)
    return str(output_path)


@task
def load_processed_training(path: Path = PROCESSED_TRAINING) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Processed training data not found at {path}. Run build_offline_features_flow first.")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["datetime"])
    return df


@task
def split_train_val(
    df: pd.DataFrame, cutoff_ts: pd.Timestamp = DEFAULT_CUTOFF_TS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, List[str], Dict[str, str]]:
    logger = get_run_logger()
    feature_cols = [col for col in df.columns if col not in {"main_aqi_t_plus_h", "hazard_t_plus_h"}]
    if "datetime" in feature_cols:
        feature_cols.remove("datetime")

    X = df[feature_cols]
    y_reg = df["main_aqi_t_plus_h"]
    y_clf = df["hazard_t_plus_h"]

    train_mask = df["datetime"] <= cutoff_ts
    val_mask = df["datetime"] > cutoff_ts

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError("Train/validation split resulted in empty sets; adjust the cutoff date.")

    logger.info("Train rows: %s | Val rows: %s", train_mask.sum(), val_mask.sum())

    train_period = {
        "start": str(df.loc[train_mask, "datetime"].min()),
        "end": str(df.loc[train_mask, "datetime"].max()),
    }
    val_period = {
        "start": str(df.loc[val_mask, "datetime"].min()),
        "end": str(df.loc[val_mask, "datetime"].max()),
    }

    return (
        X[train_mask],
        X[val_mask],
        y_reg[train_mask],
        y_reg[val_mask],
        y_clf[train_mask],
        y_clf[val_mask],
        feature_cols,
        {"train": train_period, "val": val_period},
    )


@task
def train_regressor(X_train: pd.DataFrame, y_train: pd.Series, feature_cols: List[str]):
    preprocessor = make_preprocessor(feature_cols)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


@task
def train_classifier(X_train: pd.DataFrame, y_train: pd.Series, feature_cols: List[str]):
    preprocessor = make_preprocessor(feature_cols)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


@task
def evaluate_models(
    reg_model,
    clf_model,
    X_val: pd.DataFrame,
    y_val_reg: pd.Series,
    y_val_clf: pd.Series,
) -> Dict[str, Dict[str, float]]:
    reg_pred = reg_model.predict(X_val)
    clf_pred = clf_model.predict(X_val)
    clf_proba = None
    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(X_val)
        if proba.shape[1] > 1:
            clf_proba = proba[:, 1]

    reg_metrics = evaluate_regression(y_val_reg, reg_pred)
    clf_metrics = evaluate_classification(y_val_clf, clf_pred, clf_proba)
    return {"regression": reg_metrics, "classification": clf_metrics}


@task
def persist_best_models(
    reg_model,
    clf_model,
    metrics: Dict[str, Dict[str, float]],
    feature_cols: List[str],
    horizon_hours: int,
    periods: Dict[str, Dict[str, str]],
) -> None:
    PRODUCTION_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    reg_path = PRODUCTION_MODELS_DIR / "regressor_random_forest.pkl"
    clf_path = PRODUCTION_MODELS_DIR / "classifier_random_forest.pkl"
    joblib.dump(reg_model, reg_path)
    joblib.dump(clf_model, clf_path)

    metadata = {
        "model_version": "0.1.0",
        "horizon_hours": horizon_hours,
        "feature_columns": feature_cols,
        "train_period": periods.get("train"),
        "val_period": periods.get("val"),
        "regression_metrics": metrics.get("regression"),
        "classification_metrics": metrics.get("classification"),
    }

    with open(PRODUCTION_MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger = get_run_logger()
    logger.info("Saved production regressor to %s", reg_path)
    logger.info("Saved production classifier to %s", clf_path)
    logger.info("Metadata: %s", json.dumps(metadata, indent=2))


@flow
def train_models_flow(
    processed_path: Path = PROCESSED_TRAINING,
    horizon_hours: int = DEFAULT_HORIZON,
    cutoff_ts: pd.Timestamp = DEFAULT_CUTOFF_TS,
) -> Dict[str, Dict[str, float]]:
    logger = get_run_logger()
    df = load_processed_training(processed_path)
    (
        X_train,
        X_val,
        y_train_reg,
        y_val_reg,
        y_train_clf,
        y_val_clf,
        feature_cols,
        periods,
    ) = split_train_val(df, cutoff_ts)

    reg_model = train_regressor(X_train, y_train_reg, feature_cols)
    clf_model = train_classifier(X_train, y_train_clf, feature_cols)

    metrics = evaluate_models(reg_model, clf_model, X_val, y_val_reg, y_val_clf)
    logger.info("Validation metrics: %s", metrics)

    persist_best_models(reg_model, clf_model, metrics, feature_cols, horizon_hours, periods)
    return metrics


@task
def load_test_data(path: Path = TEST_FILE, horizon_hours: int = DEFAULT_HORIZON) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Test data not found at {path}")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["datetime"])
    df["city"] = df["city"].astype(str)
    df = df.drop_duplicates(subset=["city", "datetime"]).sort_values(["city", "datetime"])
    df = add_future_targets(df, horizon_hours=horizon_hours)
    df = build_features(df)
    return df


@task
def load_production_models():
    reg_path = PRODUCTION_MODELS_DIR / "regressor_random_forest.pkl"
    clf_path = PRODUCTION_MODELS_DIR / "classifier_random_forest.pkl"
    metadata_path = PRODUCTION_MODELS_DIR / "metadata.json"

    if not reg_path.exists() or not clf_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Production models/metadata not found. Run train_models_flow first.")

    reg_model = joblib.load(reg_path)
    clf_model = joblib.load(clf_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return reg_model, clf_model, metadata


@task
def evaluate_on_test(
    reg_model,
    clf_model,
    metadata: Dict,
    df: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    feature_cols: List[str] = metadata.get("feature_columns", [])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns on test set: {missing}")

    X_test = df[feature_cols]
    y_test_reg = df["main_aqi_t_plus_h"]
    y_test_clf = df["hazard_t_plus_h"]

    reg_pred = reg_model.predict(X_test)
    clf_pred = clf_model.predict(X_test)

    clf_proba = None
    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(X_test)
        if proba.shape[1] > 1:
            clf_proba = proba[:, 1]

    reg_metrics = evaluate_regression(y_test_reg, reg_pred)
    clf_metrics = evaluate_classification(y_test_clf, clf_pred, clf_proba)
    return {"regression": reg_metrics, "classification": clf_metrics}


@flow
def evaluate_on_test_flow(test_path: Path = TEST_FILE, horizon_hours: int = DEFAULT_HORIZON) -> Dict[str, Dict[str, float]]:
    logger = get_run_logger()
    df = load_test_data(test_path, horizon_hours=horizon_hours)
    reg_model, clf_model, metadata = load_production_models()
    metrics = evaluate_on_test(reg_model, clf_model, metadata, df)
    logger.info("Test metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    # Convenience: run build + train when executed directly.
    build_offline_features_flow()
    train_models_flow()
