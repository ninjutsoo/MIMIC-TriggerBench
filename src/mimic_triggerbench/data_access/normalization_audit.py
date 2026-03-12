"""Utilities to audit normalization coverage on real MIMIC data (Phase 3).

This module is intentionally lightweight and fast:
- reads only a few columns
- supports sampling the first N rows per table
- produces a markdown report that surfaces high-frequency unmapped terms
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow.parquet as pq

from mimic_triggerbench.config import DataBackend, Settings
from mimic_triggerbench.mimic_tables import TABLE_SPECS, resolve_table_path

from .codebook_loader import load_all_codebooks
from .normalizer import Normalizer


@dataclass(frozen=True)
class TableScanResult:
    table: str
    rows_scanned: int
    mapped: int
    unmapped: int
    ambiguous: int
    top_unmapped_itemids: list[tuple[int, int]]
    top_unmapped_labels: list[tuple[str, int]]


_AUDIT_TABLES = ("labevents", "chartevents", "inputevents", "procedureevents", "prescriptions", "emar")


def _resolve_file_path(settings: Settings, table: str) -> tuple[Path, str]:
    if settings.mimic_root is None:
        raise ValueError("Settings.mimic_root is required for file backend.")
    resolved = resolve_table_path(Path(settings.mimic_root), table)
    if resolved is None:
        spec = TABLE_SPECS[table]
        raise FileNotFoundError(
            f"{table}: no table file found under {settings.mimic_root!s}; "
            f"tried {', '.join(spec.candidate_paths)}"
        )
    return resolved.path, resolved.file_format


def _read_first_rows(path: Path, file_format: str, usecols: Iterable[str], nrows: int) -> pd.DataFrame:
    # We intentionally read only the first nrows for speed and determinism.
    if file_format == "parquet":
        df = pd.read_parquet(path, columns=list(usecols))
        return df.head(nrows)
    return pd.read_csv(path, compression="infer", usecols=list(usecols), nrows=nrows)


def _cols_present(path: Path, file_format: str, candidates: list[str]) -> list[str]:
    # Read only the header row to see which columns exist.
    if file_format == "parquet":
        present_cols = set(pq.ParquetFile(path).schema.names)
    else:
        header = pd.read_csv(path, compression="infer", nrows=0)
        present_cols = set(header.columns)
    present = [c for c in candidates if c in present_cols]
    return present


def _scan_table_files_backend(
    settings: Settings,
    normalizer: Normalizer,
    table: str,
    max_rows: int,
    top_k: int,
) -> TableScanResult:
    path, file_format = _resolve_file_path(settings, table)

    if table in {"labevents", "chartevents", "inputevents", "procedureevents"}:
        candidates = ["itemid", "label", "valuenum", "valueuom", "amount", "amountuom", "rate", "rateuom"]
        cols = _cols_present(path, file_format, candidates)
        if "itemid" not in cols:
            raise ValueError(f"{table}: expected 'itemid' column in {path}")
        df = _read_first_rows(path, file_format, cols, nrows=max_rows)

        for row in df.itertuples(index=False):
            itemid = getattr(row, "itemid", None)
            label = getattr(row, "label", None) if "label" in df.columns else None
            # Prefer numeric value columns when present; otherwise None.
            value = getattr(row, "valuenum", None) if "valuenum" in df.columns else None
            unit = getattr(row, "valueuom", None) if "valueuom" in df.columns else None
            if value is None and "amount" in df.columns:
                value = getattr(row, "amount", None)
                unit = getattr(row, "amountuom", None) if "amountuom" in df.columns else unit
            if value is None and "rate" in df.columns:
                value = getattr(row, "rate", None)
                unit = getattr(row, "rateuom", None) if "rateuom" in df.columns else unit

            normalizer.normalize(source_table=table, itemid=int(itemid) if pd.notna(itemid) else None, label=label, value=value, unit=unit)

        s = normalizer.stats
        return TableScanResult(
            table=table,
            rows_scanned=len(df),
            mapped=s.mapped,
            unmapped=s.unmapped,
            ambiguous=s.ambiguous,
            top_unmapped_itemids=[(int(i), int(c)) for i, c in s.unmapped_itemids.most_common(top_k)],
            top_unmapped_labels=[(str(l), int(c)) for l, c in s.unmapped_labels.most_common(top_k)],
        )

    if table in {"prescriptions", "emar"}:
        candidates = ["drug", "medication", "medication_name", "medication_description", "drug_name"]
        cols = _cols_present(path, file_format, candidates)
        if not cols:
            raise ValueError(f"{table}: expected a drug/medication text column in {path}")
        df = _read_first_rows(path, file_format, cols, nrows=max_rows)
        col = cols[0]
        for v in df[col].astype("string").fillna("").tolist():
            label = str(v)
            if not label:
                continue
            normalizer.normalize(source_table=table, itemid=None, label=label)

        s = normalizer.stats
        return TableScanResult(
            table=table,
            rows_scanned=len(df),
            mapped=s.mapped,
            unmapped=s.unmapped,
            ambiguous=s.ambiguous,
            top_unmapped_itemids=[(int(i), int(c)) for i, c in s.unmapped_itemids.most_common(top_k)],
            top_unmapped_labels=[(str(l), int(c)) for l, c in s.unmapped_labels.most_common(top_k)],
        )

    raise ValueError(f"Unsupported table for scan: {table}")


def scan_normalization_coverage(
    settings: Settings,
    *,
    tables: Optional[list[str]] = None,
    max_rows_per_table: int = 200_000,
    top_k: int = 50,
) -> list[TableScanResult]:
    """Scan real MIMIC tables and compute normalization coverage statistics.

    This is meant as an *audit* step, not a strict pass/fail gate, because
    overall unmapped rate will be high until codebooks are expanded.
    """
    if tables is None:
        tables = list(_AUDIT_TABLES)

    codebooks = load_all_codebooks()
    normalizer = Normalizer(codebooks)

    results: list[TableScanResult] = []
    for table in tables:
        normalizer.reset_stats()
        if settings.backend == DataBackend.FILES:
            results.append(
                _scan_table_files_backend(
                    settings=settings,
                    normalizer=normalizer,
                    table=table,
                    max_rows=max_rows_per_table,
                    top_k=top_k,
                )
            )
        else:
            raise NotImplementedError(
                "Normalization coverage scan currently supports only the file backend. "
                "Add a Postgres scanner when needed."
            )
    return results


def write_normalization_coverage_report(
    results: list[TableScanResult],
    out_path: Path,
    *,
    settings: Settings,
    max_rows_per_table: int,
    top_k: int,
) -> None:
    lines: list[str] = []
    lines.append("# Normalization coverage (generated)")
    lines.append("")
    lines.append(f"- Backend: {settings.backend.value}")
    lines.append(f"- Root: `{settings.mimic_root}`" if settings.mimic_root else "- Root: (n/a)")
    lines.append(f"- Max rows scanned per table: {max_rows_per_table}")
    lines.append(f"- Top-K unmapped shown: {top_k}")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| table | rows_scanned | mapped | unmapped | ambiguous | mapped_rate |")
    lines.append("|-------|------------:|------:|--------:|----------:|-----------:|")
    for r in results:
        denom = max(r.rows_scanned, 1)
        rate = r.mapped / denom
        lines.append(
            f"| `{r.table}` | {r.rows_scanned} | {r.mapped} | {r.unmapped} | {r.ambiguous} | {rate:.3f} |"
        )
    lines.append("")

    for r in results:
        lines.append(f"## `{r.table}`")
        lines.append("")
        lines.append(f"- Rows scanned: {r.rows_scanned}")
        lines.append(f"- Mapped: {r.mapped}")
        lines.append(f"- Unmapped: {r.unmapped}")
        lines.append(f"- Ambiguous (mapped but flagged): {r.ambiguous}")
        lines.append("")

        if r.top_unmapped_itemids:
            lines.append("### Top unmapped itemids")
            lines.append("")
            for iid, c in r.top_unmapped_itemids:
                lines.append(f"- `itemid={iid}`: {c}")
            lines.append("")

        if r.top_unmapped_labels:
            lines.append("### Top unmapped labels")
            lines.append("")
            for lbl, c in r.top_unmapped_labels:
                lines.append(f"- `{lbl}`: {c}")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
