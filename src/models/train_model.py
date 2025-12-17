"""Training helpers for AQI forecasting and hazard classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import MODELS_DIR, RAW_TRAINING_DIR
from src.features.build_features import build_features
from src.features.targets import add_future_targets

TRAINING_FILE = RAW_TRAINING_DIR / "training_all_cities_until_2024_06_30.parquet"
TRAINING_SAMPLE_FILE = RAW_TRAINING_DIR / "training_sample.csv"
HORIZON_HOURS = 3
CUTOFF_TS = pd.Timestamp("2023-12-31 23:59:59")


def load_and_prepare_data(training_path: Path = TRAINING_FILE, horizon_hours: int = HORIZON_HOURS) -> pd.DataFrame:
    """Load raw training data, add targets, and build features.

    Falls back to a lightweight CSV sample if the full parquet is unavailable.
    """
    path_to_use = training_path if training_path.exists() else TRAINING_SAMPLE_FILE
    if not path_to_use.exists():
        raise FileNotFoundError(
            f"Training data not found. Expected {training_path} or fallback sample at {TRAINING_SAMPLE_FILE}."
        )

    if path_to_use.suffix == ".parquet":
        df = pd.read_parquet(path_to_use)
    else:
        df = pd.read_csv(path_to_use)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["datetime"])
    df["city"] = df["city"].astype(str)
    df = df.drop_duplicates(subset=["city", "datetime"]).sort_values(["city", "datetime"])

    df = add_future_targets(df, horizon_hours=horizon_hours)
    df = build_features(df)
    return df


def make_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""
    cat_features = [col for col in feature_columns if col == "city"]
    numeric_features = [col for col in feature_columns if col not in cat_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if cat_features:
        transformers.append(("cat", categorical_transformer, cat_features))

    return ColumnTransformer(transformers=transformers)


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """Compute basic regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return {"mse": float(mse), "rmse": rmse, "mae": float(mae)}


def evaluate_classification(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, object]:
    """Compute basic classification metrics with optional ROC-AUC."""
    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def prepare_train_val_split(
    df: pd.DataFrame, cutoff_ts: pd.Timestamp = CUTOFF_TS
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Prepare train/val splits and feature columns using the baseline training logic.
    Returns X_train, y_train_reg, y_train_clf, X_val, y_val_reg, y_val_clf, feature_columns.
    """
    feature_columns = [col for col in df.columns if col not in {"main_aqi_t_plus_h", "hazard_t_plus_h"}]
    if "datetime" in feature_columns:
        feature_columns.remove("datetime")

    X = df[feature_columns]
    y_reg = df["main_aqi_t_plus_h"]
    y_clf = df["hazard_t_plus_h"]

    train_mask = df["datetime"] <= cutoff_ts
    val_mask = df["datetime"] > cutoff_ts

    X_train, X_val = X[train_mask], X[val_mask]
    y_train_reg, y_val_reg = y_reg[train_mask], y_reg[val_mask]
    y_train_clf, y_val_clf = y_clf[train_mask], y_clf[val_mask]

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Train/validation split resulted in empty sets; adjust the cutoff date.")

    return X_train, y_train_reg, y_train_clf, X_val, y_val_reg, y_val_clf, feature_columns


def fit_baseline_models(
    X_train: pd.DataFrame,
    y_train_reg: pd.Series,
    y_train_clf: pd.Series,
    feature_columns: List[str],
) -> tuple[Pipeline, Pipeline]:
    """Fit the baseline RandomForest regressor and classifier with preprocessing."""
    preprocessor_reg = make_preprocessor(feature_columns)
    preprocessor_clf = make_preprocessor(feature_columns)

    reg_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor_reg),
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

    clf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor_clf),
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

    reg_model.fit(X_train, y_train_reg)
    clf_model.fit(X_train, y_train_clf)
    return reg_model, clf_model


def train_models(df: pd.DataFrame, cutoff_ts: pd.Timestamp = CUTOFF_TS) -> dict:
    """Train baseline models and return fitted estimators plus metrics/metadata."""
    (
        X_train,
        y_train_reg,
        y_train_clf,
        X_val,
        y_val_reg,
        y_val_clf,
        feature_columns,
    ) = prepare_train_val_split(df, cutoff_ts=cutoff_ts)

    reg_model, clf_model = fit_baseline_models(X_train, y_train_reg, y_train_clf, feature_columns)

    reg_pred = reg_model.predict(X_val)
    clf_pred = clf_model.predict(X_val)

    clf_proba = None
    if hasattr(clf_model, "predict_proba"):
        proba = clf_model.predict_proba(X_val)
        if proba.shape[1] > 1:
            clf_proba = proba[:, 1]

    reg_metrics = evaluate_regression(y_val_reg, reg_pred)
    clf_metrics = evaluate_classification(y_val_clf, clf_pred, clf_proba)

    return {
        "reg_model": reg_model,
        "clf_model": clf_model,
        "reg_metrics": reg_metrics,
        "clf_metrics": clf_metrics,
        "feature_columns": feature_columns,
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
    }


def save_outputs(models: dict, horizon_hours: int = HORIZON_HOURS) -> None:
    """Persist models and metadata to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    reg_path = MODELS_DIR / "regressor_random_forest.pkl"
    clf_path = MODELS_DIR / "classifier_random_forest.pkl"
    joblib.dump(models["reg_model"], reg_path)
    joblib.dump(models["clf_model"], clf_path)

    metadata = {
        "model_version": "0.1.0",
        "horizon_hours": horizon_hours,
        "feature_columns": models["feature_columns"],
        "train_rows": models["train_rows"],
        "val_rows": models["val_rows"],
        "regression_metrics": models["reg_metrics"],
        "classification_metrics": models["clf_metrics"],
    }

    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved regression model to", reg_path)
    print("Saved classification model to", clf_path)
    print("Metadata:", json.dumps(metadata, indent=2))


def main() -> None:
    df = load_and_prepare_data()
    outputs = train_models(df)
    save_outputs(outputs)


if __name__ == "__main__":
    main()
