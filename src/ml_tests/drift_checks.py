from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

REFERENCE_PATH = Path("data/reference_stats.json")


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Reference stats not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def psi(ref, cur):
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    epsilon = 1e-6
    return np.sum((ref - cur) * np.log((ref + epsilon) / (cur + epsilon)))


def run_drift_checks(sample_df: pd.DataFrame, threshold: float = 0.3) -> None:
    ref = load_reference()
    for col, stats in ref.items():
        if col not in sample_df.columns:
            raise ValueError(f"Missing column for drift check: {col}")
        bins = stats["bins"]
        ref_dist = np.asarray(stats["ref"])
        counts, _ = np.histogram(sample_df[col], bins=bins)
        cur_dist = counts / max(counts.sum(), 1)
        score = psi(ref_dist, cur_dist)
        ks_stat = ks_2samp(
            np.random.choice(len(bins) - 1, size=100, p=ref_dist),
            np.random.choice(len(bins) - 1, size=100, p=cur_dist if cur_dist.sum() > 0 else ref_dist),
        ).statistic
        if score > threshold or ks_stat > 0.5:
            raise ValueError(f"Drift detected on {col}: PSI={score:.3f}, KS={ks_stat:.3f}")
