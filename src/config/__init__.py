"""Project-wide configuration and common paths."""

from pathlib import Path

from pydantic_settings import BaseSettings

# BASE_DIR is two levels up from this file (repo root)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RAW_TRAINING_DIR = DATA_DIR / "raw" / "training"
RAW_TESTING_DIR = DATA_DIR / "raw" / "testing"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
PRODUCTION_MODELS_DIR = MODELS_DIR / "production"


class Settings(BaseSettings):
    """Environment-driven settings container."""

    environment: str = "dev"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
