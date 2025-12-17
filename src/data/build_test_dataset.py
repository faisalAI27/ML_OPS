"""Build a unified testing parquet across all cities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from src.config import RAW_TESTING_DIR, RAW_TRAINING_DIR
from src.data.build_training_dataset import OUTPUT_FILENAME as TRAINING_OUTPUT_FILENAME

CITY_FILES = {
    "islamabad": "islamabad_complete_data_testing_july_to_dec_2024.csv",
    "lahore": "lahore_complete_data_testing_july_to_dec_2024.csv",
    "karachi": "karachi_complete_data_testing_july_to_dec_2024.csv",
    "peshawar": "peshawar_complete_data_testing_july_to_dec_2024.csv",
    "quetta": "quetta_complete_data_testing_july_to_dec_2024.csv",
}

OUTPUT_FILENAME = "testing_all_cities_2024_07_to_12.parquet"
DEFAULT_TRAINING_REFERENCE = RAW_TRAINING_DIR / TRAINING_OUTPUT_FILENAME


def standardize_columns(columns: Iterable[str]) -> List[str]:
    return [col.strip().replace(".", "_") for col in columns]


def read_city_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_city_dataset(city: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected data for {city} at {path}")

    df = read_city_file(path)
    df.columns = standardize_columns(df.columns)

    if "datetime" not in df.columns:
        raise ValueError(f"`datetime` column not found in {path}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["datetime"])
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["city"] = city
    return df


def align_columns(df: pd.DataFrame, expected: List[str], city: str) -> pd.DataFrame:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in dataset for {city}")

    extras = [col for col in df.columns if col not in expected + ["city"]]
    if extras:
        df = df.drop(columns=extras)

    return df[expected + ["city"]]


def load_expected_columns(reference_path: Path) -> List[str] | None:
    if not reference_path.exists():
        return None

    try:
        import pyarrow.parquet as pq

        return [name for name in pq.ParquetFile(reference_path).schema.names if name != "city"]
    except Exception:
        df = pd.read_parquet(reference_path)
        return [col for col in df.columns if col != "city"]


def build_test_dataset(
    source_dir: Path | None = None,
    output_path: Path | None = None,
    training_reference: Path | None = None,
) -> Path:
    source_root = Path(source_dir) if source_dir else RAW_TESTING_DIR
    output = Path(output_path) if output_path else RAW_TESTING_DIR / OUTPUT_FILENAME
    training_reference = (
        Path(training_reference) if training_reference else DEFAULT_TRAINING_REFERENCE
    )

    frames: list[pd.DataFrame] = []
    expected_columns = load_expected_columns(training_reference)

    for city, filename in CITY_FILES.items():
        city_path = source_root / filename
        df = load_city_dataset(city, city_path)

        if expected_columns is None:
            expected_columns = [col for col in df.columns if col != "city"]
        else:
            df = align_columns(df, expected_columns, city)

        frames.append(df[expected_columns + ["city"]])

    combined = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine city-wise testing data into a single parquet.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=RAW_TESTING_DIR,
        help="Directory containing the city testing files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_TESTING_DIR / OUTPUT_FILENAME,
        help="Output parquet path.",
    )
    parser.add_argument(
        "--training-reference",
        type=Path,
        default=DEFAULT_TRAINING_REFERENCE,
        help="Path to the training parquet to enforce matching columns (if available).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_test_dataset(
        source_dir=args.source_dir,
        output_path=args.output,
        training_reference=args.training_reference,
    )
    print(f"Saved combined testing data to {output_path}")


if __name__ == "__main__":
    main()
