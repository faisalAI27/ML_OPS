import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.recommendations import build_recommendations  # noqa: E402


@pytest.mark.parametrize("aqi,hazard", [(1, "Safe"), (3.2, "Safe / Moderate"), (4.7, "Hazardous")])
def test_recommendations_structure(aqi, hazard):
    recs = build_recommendations(aqi_3h=aqi, hazard_label=hazard, current_aqi=3.0, horizon_hours=3, features=None)
    assert isinstance(recs, dict)
    for key in ["severity_bucket", "headline", "summary", "actions_general", "actions_sensitive_groups", "based_on"]:
        assert key in recs
    assert isinstance(recs["actions_general"], list)
    assert isinstance(recs["actions_sensitive_groups"], list)
    assert isinstance(recs["based_on"], dict)


def test_recommendations_no_crash_all_aqi():
    # Ensure no crash for AQI 1-5 boundaries
    for aqi in [1, 2, 3, 4, 5]:
        recs = build_recommendations(aqi_3h=aqi, hazard_label="Safe", features=None)
        assert recs["severity_bucket"] is not None


def test_recommendations_ml_fallback(monkeypatch):
    import app.recommendations as recmod

    monkeypatch.setattr(recmod, "recommender_model", None)
    recs = recmod.build_recommendations(aqi_3h=4.5, hazard_label="Hazardous", features=None)
    assert recs["severity_bucket"]


def test_recommendations_ml_path(monkeypatch):
    import app.recommendations as recmod

    class DummyRec:
        def predict(self, X):
            return [3]

    monkeypatch.setattr(recmod, "recommender_model", DummyRec())
    monkeypatch.setattr(recmod, "recommender_path", "dummy.pkl")
    import pandas as pd

    sample = pd.DataFrame({"f1": [1]})
    recs = recmod.build_recommendations(aqi_3h=2.0, hazard_label="Safe", features=sample)
    assert recs["severity_bucket"] == "Unhealthy"
    assert recs["based_on"]["recommender_model"] == "dummy.pkl"
