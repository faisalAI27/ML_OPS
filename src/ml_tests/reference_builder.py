import json
from pathlib import Path
import numpy as np
import pandas as pd


def compute_reference_stats(df: pd.DataFrame, cols: list[str], bins: int = 5) -> dict:
    stats = {}
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
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
