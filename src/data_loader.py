"""Local-Only Data Loader for Vocational Operations Analytics & ML Project.

Reads strictly from anonymized local CSV files inside `data/` — zero external
cloud API calls, zero credentials, and zero PII.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def load_maintenance_data(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load anonymized machine maintenance & IoT proxy workload dataset."""
    path = Path(data_dir) / "maintenance_anonymized.csv"
    df = pd.read_csv(path)
    for col in ("tanggal_beli", "tanggal_servis", "tanggal_servis_sebelumnya"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def load_dropout_data(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load anonymized student enrollment, attendance, and drop-out dataset."""
    path = Path(data_dir) / "dropout_anonymized.csv"
    return pd.read_csv(path)


def load_dropout_reasons_data(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load anonymized qualitative drop-out reasons dataset."""
    path = Path(data_dir) / "dropout_reasons_anonymized.csv"
    return pd.read_csv(path)


def load_curriculum_tracer_data(data_dir: Path | str = DATA_DIR) -> pd.DataFrame:
    """Load anonymized curriculum module scores & alumni tracer study dataset."""
    path = Path(data_dir) / "curriculum_tracer_anonymized.csv"
    return pd.read_csv(path)


def load_all_local_datasets(data_dir: Path | str = DATA_DIR) -> Dict[str, pd.DataFrame]:
    """Load all 4 anonymized local datasets as a dictionary."""
    return {
        "maintenance": load_maintenance_data(data_dir),
        "dropout": load_dropout_data(data_dir),
        "dropout_reasons": load_dropout_reasons_data(data_dir),
        "curriculum_tracer": load_curriculum_tracer_data(data_dir),
    }
