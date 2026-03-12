from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine

from mimic_triggerbench.config import Settings, DataBackend
from mimic_triggerbench.mimic_tables import TABLE_SPECS, TableSpec, resolve_table_path


def _missing_required_columns(columns: Iterable[str], required: Iterable[str]) -> list[str]:
    seen = set(columns)
    return [c for c in required if c not in seen]


def _validate_table_schema(df: pd.DataFrame, spec: TableSpec, table_name: str, source: str) -> None:
    missing = _missing_required_columns(df.columns, spec.required_columns)
    if missing:
        raise ValueError(
            f"Table {table_name!r} from {source!r} is missing required columns: "
            + ", ".join(repr(c) for c in missing)
        )


def load_table_dataframe(settings: Settings, table: str) -> pd.DataFrame:
    """Load a required MIMIC-IV table into a pandas DataFrame.

    Phase 1 helper to make raw tables programmatically accessible.
    """
    table = table.lower()
    spec = TABLE_SPECS.get(table)
    if spec is None:
        raise KeyError(f"Unknown table for required MIMIC access: {table!r}")

    if settings.backend == DataBackend.FILES:
        if settings.mimic_root is None:
            raise ValueError("mimic_root must be set for file backend.")
        resolved = resolve_table_path(Path(settings.mimic_root), table)
        if resolved is None:
            candidates = ", ".join(spec.candidate_paths)
            raise FileNotFoundError(
                f"No file found for table {table!r} under {settings.mimic_root!s}. "
                f"Tried: {candidates}"
            )
        if resolved.file_format == "parquet":
            df = pd.read_parquet(resolved.path)
        else:
            df = pd.read_csv(resolved.path, compression="infer")
        _validate_table_schema(df, spec, table, str(resolved.path))
        return df

    if settings.backend == DataBackend.POSTGRES:
        if not settings.postgres_dsn:
            raise ValueError("postgres_dsn must be set for postgres backend.")
        engine = create_engine(settings.postgres_dsn)
        # We don't enforce schema name here; callers can fully qualify if needed.
        df = pd.read_sql_table(table, con=engine)
        _validate_table_schema(df, spec, table, "postgres")
        return df

    raise ValueError(f"Unsupported backend: {settings.backend}")
