import json
from pathlib import Path
import pandas as pd

from src.ml_tests.reference_builder import save_reference_stats, compute_reference_stats


def test_compute_reference_stats_and_save(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        }
    )
    stats = compute_reference_stats(df, ["a", "b"], bins=3)
    assert "a" in stats and "b" in stats
    out_path = tmp_path / "ref.json"
    save_reference_stats(df, ["a", "b"], out_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert "a" in loaded and "b" in loaded
