"""Load and reconcile the Phase 3 mapping ledger against runtime codebooks."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from .codebook_models import Codebook, CodebookEntry
from .mapping_ledger_models import MappingDecision, MappingLedgerRow, ReviewStatus

_LEDGER_FILENAME = "mapping_ledger.csv"
_LEDGER_PATH = Path(__file__).resolve().parent / "mappings" / _LEDGER_FILENAME


@dataclass(frozen=True)
class MappingLedgerReconciliation:
    total_rows: int
    decision_counts: Dict[str, int]
    issue_messages: List[str]

    @property
    def ok(self) -> bool:
        return len(self.issue_messages) == 0


def mapping_ledger_path() -> Path:
    return _LEDGER_PATH


def _normalize_value(v: object) -> object:
    if pd.isna(v):
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        return s
    return v


def _coerce_int(v: object) -> int | None:
    if v is None:
        return None
    try:
        if isinstance(v, str):
            return int(v)
        if isinstance(v, (int, float)):
            return int(v)
    except (TypeError, ValueError):
        return None
    return None


def _repair_shifted_row(rec: dict[str, object]) -> dict[str, object]:
    """Repair rows where optional `snomed_code` was omitted in CSV serialization.

    In that case, values are shifted left:
    snomed_code <- source_frequency_count
    source_frequency_count <- unit_frequency_summary
    unit_frequency_summary <- review_status
    review_status <- review_note
    review_note <- None
    """

    status_values = {s.value for s in ReviewStatus}
    freq_ok = _coerce_int(rec.get("source_frequency_count")) is not None
    status_ok = rec.get("review_status") in status_values
    if freq_ok or status_ok:
        return rec

    shifted_freq = _coerce_int(rec.get("snomed_code"))
    if shifted_freq is None:
        return rec

    repaired = dict(rec)
    repaired["source_frequency_count"] = shifted_freq
    repaired["unit_frequency_summary"] = rec.get("source_frequency_count")
    repaired["review_status"] = rec.get("unit_frequency_summary")
    repaired["review_note"] = rec.get("review_status")
    repaired["snomed_code"] = None
    return repaired


def load_mapping_ledger(path: Path | None = None) -> List[MappingLedgerRow]:
    ledger_path = path or _LEDGER_PATH
    if not ledger_path.exists():
        raise FileNotFoundError(ledger_path)

    df = pd.read_csv(ledger_path)
    rows: List[MappingLedgerRow] = []
    for rec in df.to_dict(orient="records"):
        normalized = {k: _normalize_value(v) for k, v in rec.items()}
        normalized = _repair_shifted_row(normalized)
        for key in ("source_identifier", "loinc_code", "rxnorm_code", "snomed_code"):
            v = normalized.get(key)
            if v is None or isinstance(v, str):
                continue
            if isinstance(v, float) and v.is_integer():
                normalized[key] = str(int(v))
            else:
                normalized[key] = str(v)
        rows.append(MappingLedgerRow.model_validate(normalized))
    return rows


def _build_canonical_index(codebooks: Dict[str, Codebook]) -> Dict[str, CodebookEntry]:
    idx: Dict[str, CodebookEntry] = {}
    for cb in codebooks.values():
        for entry in cb.entries:
            idx[entry.canonical_name] = entry
    return idx


def _label_matches(source_label: str, raw_patterns: Sequence[str]) -> bool:
    source = source_label.lower()
    for pattern in raw_patterns:
        p = pattern.lower()
        if fnmatch.fnmatch(source, p) or fnmatch.fnmatch(p, source):
            return True
    return False


def reconcile_mapping_ledger(
    codebooks: Dict[str, Codebook],
    rows: Iterable[MappingLedgerRow],
) -> MappingLedgerReconciliation:
    index = _build_canonical_index(codebooks)
    issues: List[str] = []
    decision_counts: Dict[str, int] = {d.value: 0 for d in MappingDecision}

    all_rows = list(rows)
    for i, row in enumerate(all_rows, start=1):
        decision_counts[row.mapping_decision.value] += 1
        mapped = row.mapping_decision in {
            MappingDecision.MAP_AS_IS,
            MappingDecision.MAP_CONVERT_UNIT,
        }
        if not mapped:
            continue

        if row.canonical_concept is None:
            issues.append(f"row {i}: mapped decision without canonical_concept")
            continue

        entry = index.get(row.canonical_concept)
        if entry is None:
            issues.append(f"row {i}: canonical_concept={row.canonical_concept!r} not found in codebooks")
            continue

        if row.source_table not in entry.source_tables:
            issues.append(
                f"row {i}: source_table={row.source_table!r} not in codebook source_tables for "
                f"{row.canonical_concept!r}"
            )

        identifier_ok = False
        try:
            identifier = int(row.source_identifier)
        except (TypeError, ValueError):
            identifier = None

        if identifier is not None:
            identifier_ok = identifier in entry.raw_itemids
        else:
            identifier_ok = _label_matches(row.source_label, entry.raw_labels)

        if not identifier_ok:
            issues.append(
                f"row {i}: source_identifier/source_label does not match codebook entry for "
                f"{row.canonical_concept!r}"
            )

        if row.mapping_decision == MappingDecision.MAP_CONVERT_UNIT:
            if entry.canonical_unit and row.target_unit and entry.canonical_unit.lower() != row.target_unit.lower():
                issues.append(
                    f"row {i}: target_unit={row.target_unit!r} does not match codebook canonical_unit="
                    f"{entry.canonical_unit!r}"
                )

            conv_match = any(
                (row.source_unit is not None and c.from_unit.lower() == row.source_unit.lower())
                and row.conversion_factor is not None
                and abs(c.factor - row.conversion_factor) < 1e-9
                and row.conversion_offset is not None
                and abs(c.offset - row.conversion_offset) < 1e-9
                for c in entry.conversions
            )
            if not conv_match:
                issues.append(
                    f"row {i}: conversion rule does not match codebook conversion for {row.canonical_concept!r}"
                )

    for required in MappingDecision:
        if decision_counts[required.value] == 0:
            issues.append(f"ledger missing required decision class: {required.value}")

    return MappingLedgerReconciliation(
        total_rows=len(all_rows),
        decision_counts=decision_counts,
        issue_messages=issues,
    )


def assert_mapping_ledger_consistent(codebooks: Dict[str, Codebook], rows: Iterable[MappingLedgerRow]) -> None:
    reconciliation = reconcile_mapping_ledger(codebooks, rows)
    if reconciliation.ok:
        return
    joined = "\n".join(f"- {msg}" for msg in reconciliation.issue_messages)
    raise ValueError(f"Mapping ledger reconciliation failed:\n{joined}")
