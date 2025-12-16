"""Run all lightweight ML checks. Exits non-zero on failure."""

import sys
import pandas as pd
from pathlib import Path

from src.ml_tests.data_integrity_checks import run_data_integrity_checks
from src.ml_tests.drift_checks import run_drift_checks
from src.ml_tests.model_behavior_checks import run_model_behavior_checks


def main():
    try:
        sample_path = Path("tests/data/current_sample.csv")
        df_sample = pd.read_csv(sample_path)

        run_data_integrity_checks(df_sample)
        run_drift_checks(df_sample)
        run_model_behavior_checks()
        print("ML checks passed.")
    except Exception as exc:
        print(f"ML checks failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
