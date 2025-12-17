"""Lightweight experiment tracker for logging training runs to CSV + JSON snapshots."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOG_PATH = Path("reports/experiments_log.csv")
LATEST_JSON = Path("reports/experiments_latest.json")


def log_experiment(
    run_id: str,
    reg_metrics: Dict[str, Any],
    clf_metrics: Dict[str, Any],
    recommender_metrics: Dict[str, Any],
    model_paths: Dict[str, str],
    notes: Optional[str] = None,
) -> None:
    """Append a single experiment record; never raises to avoid breaking callers."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    row = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "run_id": run_id,
        "reg_rmse": reg_metrics.get("rmse"),
        "reg_mae": reg_metrics.get("mae"),
        "reg_r2": reg_metrics.get("r2"),
        "clf_accuracy": clf_metrics.get("accuracy"),
        "clf_f1": clf_metrics.get("f1"),
        "clf_roc_auc": clf_metrics.get("roc_auc"),
        "rec_accuracy": recommender_metrics.get("accuracy"),
        "reg_model_path": model_paths.get("regressor"),
        "clf_model_path": model_paths.get("classifier"),
        "rec_model_path": model_paths.get("recommender"),
        "metadata_path": model_paths.get("metadata"),
        "notes": notes or "",
    }
    try:
        write_header = not LOG_PATH.exists()
        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # Also write a latest snapshot for easy artifact export in CI
        snapshot = {
            "timestamp_utc": timestamp,
            "run_id": run_id,
            "regression": reg_metrics,
            "classification": clf_metrics,
            "recommender": recommender_metrics,
            "model_paths": model_paths,
            "notes": notes or "",
        }
        with open(LATEST_JSON, "w", encoding="utf-8") as jf:
            json.dump(snapshot, jf, indent=2)
    except Exception:
        # Silent by design; tracking should not block training
        return
