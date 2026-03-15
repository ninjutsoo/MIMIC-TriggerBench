from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence

import pandas as pd
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.engine import Engine

from mimic_triggerbench.config import Settings, DataBackend
from mimic_triggerbench.mimic_tables import (
    TABLE_SPECS,
    iter_table_specs,
    preferred_required_paths,
    resolve_table_path,
)


# Backward-compatible export used by a few tests.
REQUIRED_TABLE_FILES = list(preferred_required_paths())


@dataclass(frozen=True)
class TableStatus:
    table_name: str
    present: bool
    source: str | None
    source_format: str | None
    schema_ok: bool | None
    missing_columns: tuple[str, ...] = ()
    note: str | None = None


def _missing_required_columns(actual: Iterable[str], required: Iterable[str]) -> tuple[str, ...]:
    actual_set = set(actual)
    return tuple(c for c in required if c not in actual_set)


def _read_columns(path: Path, file_format: str) -> list[str]:
    if file_format == "parquet":
        return list(pq.ParquetFile(path).schema.names)
    header = pd.read_csv(path, compression="infer", nrows=0)
    return list(header.columns)


def _check_files(root: Path) -> List[TableStatus]:
    statuses: List[TableStatus] = []
    for spec in iter_table_specs():
        resolved = resolve_table_path(root, spec.table_name)
        if resolved is None:
            statuses.append(
                TableStatus(
                    table_name=spec.table_name,
                    present=False,
                    source=None,
                    source_format=None,
                    schema_ok=None,
                    note=f"missing: tried {', '.join(spec.candidate_paths)}",
                )
            )
            continue

        cols = _read_columns(resolved.path, resolved.file_format)
        missing = _missing_required_columns(cols, spec.required_columns)
        statuses.append(
            TableStatus(
                table_name=spec.table_name,
                present=True,
                source=str(resolved.path),
                source_format=resolved.file_format,
                schema_ok=(len(missing) == 0),
                missing_columns=missing,
                note=None,
            )
        )
    return statuses


def _required_postgres_tables() -> List[str]:
    return list(TABLE_SPECS.keys())


def _postgres_table_exists(engine: "Engine", table_name: str, schemas: Sequence[str]) -> Optional[str]:
    """Return schema name if found, else None."""
    from sqlalchemy import text

    q = text(
        """
        SELECT table_schema
        FROM information_schema.tables
        WHERE table_name = :table_name
          AND table_schema = ANY(:schemas)
        LIMIT 1
        """
    )
    with engine.connect() as conn:
        row = conn.execute(q, {"table_name": table_name, "schemas": list(schemas)}).fetchone()
    return row[0] if row else None


def _postgres_table_columns(engine: "Engine", table_name: str, schema: str) -> list[str]:
    from sqlalchemy import text

    q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = :table_name
          AND table_schema = :schema
        ORDER BY ordinal_position
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(q, {"table_name": table_name, "schema": schema}).fetchall()
    return [r[0] for r in rows]


def _check_postgres(dsn: str) -> List[TableStatus]:
    # Lazy import: SQLAlchemy import can be slow on some Windows setups due to
    # platform/WMI queries during import. Only load it when Postgres backend is used.
    from sqlalchemy import create_engine

    engine = create_engine(dsn)
    # Common schema names people use for MIMIC-IV; include "public" as a fallback.
    candidate_schemas = ["mimiciv_icu", "mimiciv_hosp", "mimiciv_ed", "mimiciv_derived", "public"]
    statuses: List[TableStatus] = []
    for t in _required_postgres_tables():
        schema = _postgres_table_exists(engine, t, candidate_schemas)
        if schema:
            cols = _postgres_table_columns(engine, t, schema)
            required = TABLE_SPECS[t].required_columns
            missing = _missing_required_columns(cols, required)
            statuses.append(
                TableStatus(
                    table_name=t,
                    present=True,
                    source=f"{schema}.{t}",
                    source_format="postgres",
                    schema_ok=(len(missing) == 0),
                    missing_columns=missing,
                    note=f"found in schema `{schema}`",
                )
            )
        else:
            statuses.append(
                TableStatus(
                    table_name=t,
                    present=False,
                    source=None,
                    source_format=None,
                    schema_ok=None,
                    note=f"not found in {candidate_schemas}",
                )
            )
    return statuses


def generate_inventory_report(settings: Settings, output_path: Path) -> None:
    """Generate a simple markdown inventory report for required tables.
    """
    console = Console()
    root: Optional[Path] = None
    if settings.backend == DataBackend.FILES:
        if settings.mimic_root is None:
            raise ValueError("Settings.mimic_root is not set; cannot scan files.")
        root = settings.mimic_root
        statuses = _check_files(root)
        title = "MIMIC-IV Required Tables (file presence)"
    elif settings.backend == DataBackend.POSTGRES:
        if not settings.postgres_dsn:
            raise ValueError("Settings.postgres_dsn is not set; cannot scan Postgres.")
        statuses = _check_postgres(settings.postgres_dsn)
        title = "MIMIC-IV Required Tables (Postgres presence)"
    else:
        raise ValueError(f"Unsupported backend: {settings.backend}")

    table = Table(title=title)
    table.add_column("Table")
    table.add_column("Present")
    table.add_column("Source")
    table.add_column("Format")
    table.add_column("Schema")
    table.add_column("Missing required columns")

    missing = 0
    schema_fail = 0
    for st in statuses:
        present_str = "yes" if st.present else "no"
        if not st.present:
            missing += 1
        schema = "-"
        missing_cols = "-"
        if st.schema_ok is True:
            schema = "ok"
        elif st.schema_ok is False:
            schema = "missing cols"
            schema_fail += 1
            missing_cols = ", ".join(st.missing_columns) if st.missing_columns else "-"
        else:
            schema = "-"
        table.add_row(
            st.table_name,
            present_str,
            st.source or "-",
            st.source_format or "-",
            schema,
            missing_cols,
        )

    console.print(table)

    lines = [
        "# Data inventory (generated)",
        "",
        f"- Backend: {settings.backend.value}",
        f"- Root: `{root}`" if root else "- Root: (n/a)",
        "",
        "| table | present | source | format | schema_ok | missing_required_columns | note |",
        "|-------|---------|--------|--------|-----------|--------------------------|------|",
    ]
    for st in statuses:
        present_str = "yes" if st.present else "no"
        if st.schema_ok is True:
            schema_ok = "yes"
        elif st.schema_ok is False:
            schema_ok = "no"
        else:
            schema_ok = "-"
        missing_cols = ", ".join(st.missing_columns) if st.missing_columns else "-"
        source = f"`{st.source}`" if st.source else "-"
        note = st.note or "-"
        lines.append(
            f"| `{st.table_name}` | {present_str} | {source} | {st.source_format or '-'} | "
            f"{schema_ok} | {missing_cols} | {note} |"
        )

    lines.append("")
    lines.append(f"Missing tables: **{missing}**")
    lines.append(f"Schema-invalid tables: **{schema_fail}**")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
