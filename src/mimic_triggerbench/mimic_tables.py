from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


SUPPORTED_FILE_SUFFIXES: Tuple[str, ...] = (".parquet", ".csv.gz", ".csv")


@dataclass(frozen=True)
class TableSpec:
    table_name: str
    candidate_paths: Tuple[str, ...]
    required_columns: Tuple[str, ...]


@dataclass(frozen=True)
class ResolvedTablePath:
    table_name: str
    path: Path
    rel_path: str
    file_format: str


def _candidate_paths(module: str, table_name: str) -> Tuple[str, ...]:
    return tuple(f"{module}/{table_name}{suffix}" for suffix in SUPPORTED_FILE_SUFFIXES)


TABLE_SPECS: Dict[str, TableSpec] = {
    # ICU module
    "icustays": TableSpec(
        table_name="icustays",
        candidate_paths=_candidate_paths("icu", "icustays"),
        required_columns=("subject_id", "hadm_id", "stay_id", "intime", "outtime"),
    ),
    "chartevents": TableSpec(
        table_name="chartevents",
        candidate_paths=_candidate_paths("icu", "chartevents"),
        required_columns=("subject_id", "hadm_id", "stay_id", "charttime", "itemid"),
    ),
    "inputevents": TableSpec(
        table_name="inputevents",
        candidate_paths=_candidate_paths("icu", "inputevents"),
        required_columns=("subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid"),
    ),
    "outputevents": TableSpec(
        table_name="outputevents",
        candidate_paths=_candidate_paths("icu", "outputevents"),
        required_columns=("subject_id", "hadm_id", "stay_id", "charttime", "itemid"),
    ),
    "procedureevents": TableSpec(
        table_name="procedureevents",
        candidate_paths=_candidate_paths("icu", "procedureevents"),
        required_columns=("subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid"),
    ),
    # Hospital module
    "admissions": TableSpec(
        table_name="admissions",
        candidate_paths=_candidate_paths("hosp", "admissions"),
        required_columns=("subject_id", "hadm_id", "admittime", "dischtime"),
    ),
    "patients": TableSpec(
        table_name="patients",
        candidate_paths=_candidate_paths("hosp", "patients"),
        required_columns=("subject_id",),
    ),
    "labevents": TableSpec(
        table_name="labevents",
        candidate_paths=_candidate_paths("hosp", "labevents"),
        required_columns=("subject_id", "hadm_id", "charttime", "itemid"),
    ),
    "prescriptions": TableSpec(
        table_name="prescriptions",
        candidate_paths=_candidate_paths("hosp", "prescriptions"),
        required_columns=("subject_id", "hadm_id"),
    ),
    "emar": TableSpec(
        table_name="emar",
        candidate_paths=_candidate_paths("hosp", "emar"),
        required_columns=("subject_id", "hadm_id"),
    ),
    "pharmacy": TableSpec(
        table_name="pharmacy",
        candidate_paths=_candidate_paths("hosp", "pharmacy"),
        required_columns=("subject_id", "hadm_id"),
    ),
    "transfers": TableSpec(
        table_name="transfers",
        candidate_paths=_candidate_paths("hosp", "transfers"),
        required_columns=("subject_id", "hadm_id"),
    ),
    "diagnoses_icd": TableSpec(
        table_name="diagnoses_icd",
        candidate_paths=_candidate_paths("hosp", "diagnoses_icd"),
        required_columns=("subject_id", "hadm_id", "icd_code"),
    ),
}


def iter_table_specs() -> Iterable[TableSpec]:
    return TABLE_SPECS.values()


def required_table_names() -> Tuple[str, ...]:
    return tuple(TABLE_SPECS.keys())


def preferred_required_paths() -> Tuple[str, ...]:
    return tuple(spec.candidate_paths[0] for spec in TABLE_SPECS.values())


def resolve_table_path(root: Path, table_name: str) -> Optional[ResolvedTablePath]:
    spec = TABLE_SPECS.get(table_name.lower())
    if spec is None:
        raise KeyError(f"Unknown required table: {table_name!r}")

    for rel in spec.candidate_paths:
        path = root / rel
        if path.exists():
            if rel.endswith(".parquet"):
                file_format = "parquet"
            elif rel.endswith(".csv.gz"):
                file_format = "csv.gz"
            elif rel.endswith(".csv"):
                file_format = "csv"
            else:
                file_format = "unknown"
            return ResolvedTablePath(
                table_name=spec.table_name,
                path=path,
                rel_path=rel,
                file_format=file_format,
            )
    return None

