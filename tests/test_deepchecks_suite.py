import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# Compatibility for NumPy 2.0 where deepchecks expects np.Inf
setattr(np, "Inf", np.inf)
setattr(np, "PINF", np.inf)
setattr(np, "NINF", -np.inf)
setattr(np, "NaN", np.nan)

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, model_evaluation

from src.models.train_model import load_and_prepare_data, TRAINING_SAMPLE_FILE, make_preprocessor


SAMPLE_SIZE = 800
RANDOM_STATE = 42
FEATURE_SUBSET = [
    "main_aqi",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "shortwave_radiation",
    "hour",
    "month",
    "city",
]


def _load_sample_df(max_rows: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Load a small sample of the training data with engineered features."""
    training_path = TRAINING_SAMPLE_FILE if TRAINING_SAMPLE_FILE.exists() else None
    df = load_and_prepare_data(training_path=training_path) if training_path else load_and_prepare_data()
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)


@pytest.mark.ml
def test_deepchecks_data_integrity_suite_fast():
    df = _load_sample_df()
    feature_cols = [c for c in FEATURE_SUBSET if c in df.columns]
    cat_features = [c for c in feature_cols if c == "city"]
    ds = Dataset(df[feature_cols], label=df["hazard_t_plus_h"], cat_features=cat_features)

    result = data_integrity().run(ds)
    assert result.passed(), result.to_json()


@pytest.mark.ml
def test_deepchecks_model_evaluation_suite_fast():
    df_raw = _load_sample_df()
    min_count = df_raw["hazard_t_plus_h"].value_counts().min()
    df = (
        df_raw.groupby("hazard_t_plus_h", group_keys=False)
        .apply(lambda x: x.sample(n=min(min_count, len(x)), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )

    feature_cols = [c for c in FEATURE_SUBSET if c in df.columns]
    X = df[feature_cols]
    y_clf = df["hazard_t_plus_h"]
    cat_features = [c for c in feature_cols if c == "city"]

    train_ds = Dataset(X, label=y_clf, cat_features=cat_features)
    test_ds = Dataset(X, label=y_clf, cat_features=cat_features)

    preprocessor = make_preprocessor(feature_cols)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    n_jobs=4,
                ),
            ),
        ]
    )
    model.fit(X, y_clf)

    result = model_evaluation().run(train_ds, test_ds, model)
    assert result.passed(), result.to_json()
