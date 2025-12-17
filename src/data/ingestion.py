"""Minimal data ingestion helper for training flows.

This is intentionally simple: it copies the latest available local training file
into an ingested location so downstream steps have a stable path. If no source
data is present, it raises a clear error.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import RAW_TRAINING_DIR
from src.models.train_model import TRAINING_FILE, TRAINING_SAMPLE_FILE

INGESTED_DIR = RAW_TRAINING_DIR / "ingested"


def ingest_local_training_data() -> Path:
    """Copy the best available local training file into an ingested folder.

    Returns the ingested file path. Raises FileNotFoundError if no source exists.
    """
    source_candidates = [TRAINING_FILE, TRAINING_SAMPLE_FILE]
    source = next((p for p in source_candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            f"No training data found; expected one of {[str(p) for p in source_candidates]}"
        )

    INGESTED_DIR.mkdir(parents=True, exist_ok=True)
    dest = INGESTED_DIR / source.name
    shutil.copy(source, dest)
    return dest
