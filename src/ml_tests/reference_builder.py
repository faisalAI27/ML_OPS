import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.main import FEATURE_COLUMNS


def compute_reference_stats(df: pd.DataFrame, cols: list[str], bins: int = 5) -> dict:
    stats = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        hist, bin_edges = np.histogram(series, bins=bins)
        dist = hist / max(hist.sum(), 1)
        stats[col] = {
            "bins": bin_edges.tolist(),
            "ref": dist.tolist(),
        }
    return stats


def save_reference_stats(df: pd.DataFrame, cols: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = compute_reference_stats(df, cols)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def save_current_sample(df: pd.DataFrame, path: Path, n: int = 200, random_state: int = 42) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = df.sample(n=min(n, len(df)), random_state=random_state)
    sample = sample.copy()
    if "datetime" not in sample.columns:
        sample["datetime"] = "2024-01-01"
    cols = ["city", "datetime"] + FEATURE_COLUMNS
    sample = sample[cols]
    sample.to_csv(path, index=False)
