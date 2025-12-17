from pathlib import Path

import pandas as pd

RAW_TRAIN_DIR = Path("data/raw/training")
OUTPUT_PATH = RAW_TRAIN_DIR / "training_all_cities_until_2024_06_30.parquet"

CITY_FILES = {
    "Islamabad": "islamabad_complete_training_data.xlsx",
    "Lahore": "lahore_complete_training_data.xlsx",
    "Karachi": "karachi_complete_training_data.xlsx",
    "Peshawar": "peshawar_complete_training_data.csv",
    "Quetta": "quetta_complete_training_data.csv",
}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Replace dots with underscores for column names."""
    return df.rename(columns={col: col.replace(".", "_") for col in df.columns})


def build_training_dataset() -> pd.DataFrame:
    frames = []
    cutoff = pd.Timestamp("2024-06-30 23:00:00")

    for city, filename in CITY_FILES.items():
        path = RAW_TRAIN_DIR / filename
        if filename.endswith(".xlsx"):
            df_city = pd.read_excel(path)
        else:
            df_city = pd.read_csv(path)

        df_city = _standardize_columns(df_city)

        df_city["city"] = city
        df_city["datetime"] = pd.to_datetime(
            df_city["datetime"],
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )
        df_city = df_city.dropna(subset=["datetime"])
        df_city["datetime"] = df_city["datetime"].dt.tz_localize(None)
        df_city = df_city[df_city["datetime"] <= cutoff]

        frames.append(df_city)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.sort_values(["city", "datetime"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(OUTPUT_PATH, index=False)

    return df_all


if __name__ == "__main__":
    df = build_training_dataset()
    print("Unified training dataset saved to:", OUTPUT_PATH)
    print("Shape:", df.shape)
