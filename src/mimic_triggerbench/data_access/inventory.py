from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine

from mimic_triggerbench.config import Settings, DataBackend


REQUIRED_TABLE_FILES = [
    # ICU module
    "icu/icustays.csv.gz",
    "icu/chartevents.csv.gz",
    "icu/inputevents.csv.gz",
    "icu/outputevents.csv.gz",
    "icu/procedureevents.csv.gz",
    # Hospital module
    "hosp/admissions.csv.gz",
    "hosp/patients.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/prescriptions.csv.gz",
    "hosp/emar.csv.gz",
    "hosp/pharmacy.csv.gz",
    "hosp/transfers.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
]


@dataclass(frozen=True)
class TableStatus:
    name: str
    present: bool
    path: Path | None
    note: str | None = None


def _check_files(root: Path, rel_paths: Iterable[str]) -> List[TableStatus]:
    statuses: List[TableStatus] = []
    for rel in rel_paths:
        full = root / rel
        statuses.append(TableStatus(name=rel, present=full.exists(), path=full if full.exists() else None))
    return statuses


def _required_postgres_tables() -> List[str]:
    # MIMIC-IV uses schemas like mimiciv_hosp, mimiciv_icu in common installations.
    # We treat any of these schema/name combinations as satisfying the requirement.
    required = [
        # ICU
        "icustays",
        "chartevents",
        "inputevents",
        "outputevents",
        "procedureevents",
        # HOSP
        "admissions",
        "patients",
        "labevents",
        "prescriptions",
        "emar",
        "pharmacy",
        "transfers",
        "diagnoses_icd",
    ]
    return required


def _postgres_table_exists(engine: Engine, table_name: str, schemas: Sequence[str]) -> Optional[str]:
    """Return schema name if found, else None."""
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


def _check_postgres(dsn: str) -> List[TableStatus]:
    engine = create_engine(dsn)
    # Common schema names people use for MIMIC-IV; include "public" as a fallback.
    candidate_schemas = ["mimiciv_icu", "mimiciv_hosp", "mimiciv_ed", "mimiciv_derived", "public"]
    statuses: List[TableStatus] = []
    for t in _required_postgres_tables():
        schema = _postgres_table_exists(engine, t, candidate_schemas)
        if schema:
            statuses.append(TableStatus(name=t, present=True, path=None, note=f"found in schema `{schema}`"))
        else:
            statuses.append(TableStatus(name=t, present=False, path=None, note=f"not found in {candidate_schemas}"))
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
        statuses = _check_files(root, REQUIRED_TABLE_FILES)
        title = "MIMIC-IV Required Tables (file presence)"
    elif settings.backend == DataBackend.POSTGRES:
        if not settings.postgres_dsn:
            raise ValueError("Settings.postgres_dsn is not set; cannot scan Postgres.")
        statuses = _check_postgres(settings.postgres_dsn)
        title = "MIMIC-IV Required Tables (Postgres presence)"
    else:
        raise ValueError(f"Unsupported backend: {settings.backend}")

    table = Table(title=title)
    table.add_column("Table / path")
    table.add_column("Present")
    table.add_column("Location / note")

    missing = 0
    for st in statuses:
        present_str = "yes" if st.present else "no"
        if not st.present:
            missing += 1
        location = "-"
        if settings.backend == DataBackend.FILES:
            location = str(st.path) if st.path else "-"
        else:
            location = st.note or "-"
        table.add_row(st.name, present_str, location)

    console.print(table)

    lines = [
        "# Data inventory (generated)",
        "",
        f"- Backend: {settings.backend.value}",
        f"- Root: `{root}`" if root else "- Root: (n/a)",
        "",
        "| table | present | location |",
        "|-------|---------|----------|",
    ]
    for st in statuses:
        present_str = "yes" if st.present else "no"
        if settings.backend == DataBackend.FILES:
            loc = f"`{st.path}`" if st.path else "-"
        else:
            loc = st.note or "-"
        lines.append(f"| `{st.name}` | {present_str} | {loc} |")

    lines.append("")
    lines.append(f"Missing tables: **{missing}**")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

