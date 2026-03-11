from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine

from mimic_triggerbench.config import Settings, DataBackend
from mimic_triggerbench.data_access.inventory import REQUIRED_TABLE_FILES


_FILE_TABLE_MAP: Dict[str, str] = {
    # map canonical table names to relative paths under mimic_root
    "icustays": "icu/icustays.csv.gz",
    "chartevents": "icu/chartevents.csv.gz",
    "inputevents": "icu/inputevents.csv.gz",
    "outputevents": "icu/outputevents.csv.gz",
    "procedureevents": "icu/procedureevents.csv.gz",
    "admissions": "hosp/admissions.csv.gz",
    "patients": "hosp/patients.csv.gz",
    "labevents": "hosp/labevents.csv.gz",
    "prescriptions": "hosp/prescriptions.csv.gz",
    "emar": "hosp/emar.csv.gz",
    "pharmacy": "hosp/pharmacy.csv.gz",
    "transfers": "hosp/transfers.csv.gz",
    "diagnoses_icd": "hosp/diagnoses_icd.csv.gz",
}


def load_table_dataframe(settings: Settings, table: str) -> pd.DataFrame:
    """Load a required MIMIC-IV table into a pandas DataFrame.

    Phase 1 helper to make raw tables programmatically accessible.
    """
    table = table.lower()
    if settings.backend == DataBackend.FILES:
        if settings.mimic_root is None:
            raise ValueError("mimic_root must be set for file backend.")
        rel = _FILE_TABLE_MAP.get(table)
        if rel is None:
            raise KeyError(f"Unknown table for file backend: {table!r}")
        path = Path(settings.mimic_root) / rel
        if not path.exists():
            raise FileNotFoundError(path)
        # For now we assume CSV (optionally gzip-compressed based on extension).
        # Parquet or other formats can be added later based on config.
        return pd.read_csv(path, compression="infer")

    if settings.backend == DataBackend.POSTGRES:
        if not settings.postgres_dsn:
            raise ValueError("postgres_dsn must be set for postgres backend.")
        engine = create_engine(settings.postgres_dsn)
        # We don't enforce schema here; callers can fully qualify if needed.
        return pd.read_sql_table(table, con=engine)

    raise ValueError(f"Unsupported backend: {settings.backend}")

