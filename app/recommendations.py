"""Recommendation helper based on forecasted AQI and hazard status."""

from __future__ import annotations

from typing import Any, Dict, Optional

import joblib
import pandas as pd
from pathlib import Path

from app.logging_config import configure_logging
from app.model_loader import MODELS_DIR, PROD_DIR

logger = configure_logging().getChild("recommendations")

RECOMMENDER_PATHS = [
    PROD_DIR / "recommender.pkl",
    MODELS_DIR / "recommender.pkl",
]

try:
    for _p in RECOMMENDER_PATHS:
        if _p.exists():
            recommender_model = joblib.load(_p)
            recommender_path = _p
            logger.info("Loaded recommender model from %s", _p)
            break
    else:
        recommender_model = None
        recommender_path = None
        logger.info("Recommender model not found; using rule-based recommendations")
except Exception:
    recommender_model = None
    recommender_path = None
    logger.exception("Failed to load recommender model; falling back to rules")

def _severity_bucket_label(aqi: float) -> str:
    """Map numeric AQI (1–5) to a coarse severity label."""
    try:
        val = int(round(float(aqi)))
    except Exception:
        return "Unknown"
    val = max(1, min(5, val))
    mapping = {
        1: "Good",
        2: "Moderate",
        3: "Unhealthy for sensitive groups",
        4: "Unhealthy",
        5: "Very Unhealthy / Hazardous",
    }
    return mapping.get(val, "Unknown")


def _bucket_from_model(features: Optional[pd.DataFrame]) -> Optional[int]:
    if recommender_model is None or features is None:
        return None
    try:
        pred = recommender_model.predict(features)
        return int(pred[0])
    except Exception:
        logger.exception("Failed to run recommender model; falling back to rule-based")
        return None


def _bucket_to_label(bucket: int) -> str:
    mapping = {0: "Good", 1: "Moderate", 2: "Unhealthy for sensitive groups", 3: "Unhealthy"}
    return mapping.get(bucket, "Unknown")


def build_recommendations(
    aqi_3h: float,
    hazard_label: str,
    current_aqi: Optional[float] = None,
    horizon_hours: int = 3,
    features: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Build human-readable recommendations given forecast AQI (1–5) and hazard label.
    Returns a JSON-serializable dict to embed in API responses.
    """
    # Try ML recommender first
    bucket_pred = _bucket_from_model(features)

    if bucket_pred is not None:
        severity_bucket = _bucket_to_label(bucket_pred)
    else:
        severity_bucket = _severity_bucket_label(aqi_3h)
    hazard_lower = (hazard_label or "").lower()
    is_hazard = "hazard" in hazard_lower

    trend_note = None
    if current_aqi is not None:
        try:
            cur = float(current_aqi)
            diff = aqi_3h - cur
            if abs(diff) < 0.2:
                trend_note = "Similar to current conditions."
            elif diff > 0:
                trend_note = "Air quality is expected to worsen."
            else:
                trend_note = "Air quality is expected to improve slightly."
        except Exception:
            trend_note = None

    if is_hazard:
        headline = f"Air quality is hazardous in the next {horizon_hours} hours."
        summary = "Limit outdoor exposure. Sensitive groups should remain indoors and use masks if going outside."
        actions_general = [
            "Avoid outdoor exercise; prefer indoor activities.",
            "Keep windows closed during peak pollution.",
            "Use an air purifier if available.",
            "Wear a well-fitted mask (e.g., N95) if you must go outside.",
        ]
        actions_sensitive = [
            "Children, elderly, pregnant women: stay indoors as much as possible.",
            "People with heart or lung disease: keep medication handy and monitor symptoms.",
            "Consult a doctor if breathing becomes difficult or if symptoms worsen.",
        ]
    else:
        headline = f"Air quality is acceptable in the next {horizon_hours} hours."
        summary = "Outdoor activity is generally fine. Stay aware if you are sensitive to pollution."
        actions_general = [
            "Normal outdoor activities are okay.",
            "Keep hydrated and prefer less congested areas for exercise.",
            "Monitor updates if conditions change.",
        ]
        actions_sensitive = [
            "Sensitive groups: consider shorter outdoor stays if you notice irritation.",
            "Keep rescue inhalers or required medication nearby if advised by your doctor.",
            "Avoid busy roads during rush hours to reduce exposure.",
        ]

    if trend_note and trend_note not in summary:
        summary = f"{summary} {trend_note}"

    return {
        "severity_bucket": severity_bucket,
        "headline": headline,
        "summary": summary,
        "actions_general": actions_general,
        "actions_sensitive_groups": actions_sensitive,
        "based_on": {
            "horizon_hours": horizon_hours,
            "predicted_aqi": float(aqi_3h),
            "current_aqi": float(current_aqi) if current_aqi is not None else None,
            "recommender_model": str(recommender_path) if recommender_path else None,
        },
    }
