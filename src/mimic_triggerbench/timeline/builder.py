"""Canonical timeline builder (Phase 4).

Merges normalized events from required MIMIC-IV tables into a single
time-ordered :class:`CanonicalEvent` sequence per ICU stay.

Direct-quote anchors implemented
---------------------------------
- ``code: [PROCEDURE, START, col(itemid)]`` / ``[PROCEDURE, END, col(itemid)]``
- ``code: [INFUSION_START, col(itemid)]`` / ``[INFUSION_END, col(itemid)]``
- ``order_id: orderid`` / ``link_order_id: linkorderid``
- ``uuid_generate_v5(...)`` via :func:`timeline.uid.generate_event_uid`
- ``LEAD(vasotime, 1) OVER (...)`` not directly applicable in pandas but the
  interval end-time is preserved from the source ``endtime`` column.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from mimic_triggerbench.config import Settings
from mimic_triggerbench.data_access.codebook_loader import load_all_codebooks
from mimic_triggerbench.data_access.normalizer import Normalizer

from .models import CanonicalEvent, events_sorted, generate_event_uid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source-table adapters
# ---------------------------------------------------------------------------

def _safe_ts(val: object) -> Optional[pd.Timestamp]:
    """Coerce to Timestamp or None, tolerating NaT / missing."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    try:
        ts = pd.Timestamp(val)
        return None if pd.isna(ts) else ts
    except Exception:
        return None


def _metadata_blob(**kwargs: object) -> str:
    clean = {k: v for k, v in kwargs.items() if v is not None and v == v}
    return json.dumps(clean, default=str)


def _build_labevents(
    df: pd.DataFrame,
    normalizer: Normalizer,
    stay_lookup: dict[tuple[int, int], int],
) -> list[CanonicalEvent]:
    """labevents → lab events.  Requires join to icustays for stay_id."""
    events: list[CanonicalEvent] = []
    dup_counter: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        subject_id = int(row.subject_id)
        hadm_id = int(row.hadm_id) if pd.notna(getattr(row, "hadm_id", None)) else 0
        ct = _safe_ts(row.charttime)
        if ct is None:
            continue

        stay_id = stay_lookup.get((subject_id, hadm_id), 0)
        if stay_id == 0:
            continue

        itemid = int(row.itemid) if pd.notna(row.itemid) else None
        label = getattr(row, "label", None)
        valuenum = float(row.valuenum) if pd.notna(getattr(row, "valuenum", None)) else None
        valueuom = str(row.valueuom) if pd.notna(getattr(row, "valueuom", None)) else None

        nr = normalizer.normalize("labevents", itemid, label, valuenum, valueuom)
        if nr is None:
            continue

        dup_key = f"labevents|{stay_id}|{ct.isoformat()}|{nr.canonical_name}|{itemid}"
        dup_counter[dup_key] += 1
        extra = str(dup_counter[dup_key] - 1) if dup_counter[dup_key] > 1 else ""

        uid = generate_event_uid(
            "labevents", stay_id, ct.to_pydatetime(),
            nr.canonical_name, raw_id=str(itemid) if itemid else None,
            extra_discriminator=extra,
        )
        events.append(CanonicalEvent(
            subject_id=subject_id,
            hadm_id=hadm_id,
            stay_id=stay_id,
            event_time=ct.to_pydatetime(),
            event_uid=uid,
            source_table="labevents",
            event_category="lab",
            canonical_name=nr.canonical_name,
            value_num=nr.normalized_value,
            unit=nr.canonical_unit,
            raw_id=str(itemid) if itemid else None,
            raw_label=nr.raw_label,
            metadata_json=_metadata_blob(
                original_value=nr.original_value,
                original_unit=valueuom,
                is_ambiguous=nr.is_ambiguous,
                codebook_version="v0.1",
            ),
        ))
    return events


def _build_chartevents(
    df: pd.DataFrame,
    normalizer: Normalizer,
) -> list[CanonicalEvent]:
    """chartevents → vital events."""
    events: list[CanonicalEvent] = []
    dup_counter: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        ct = _safe_ts(row.charttime)
        if ct is None:
            continue

        itemid = int(row.itemid) if pd.notna(row.itemid) else None
        label = getattr(row, "label", None)
        valuenum = float(row.valuenum) if pd.notna(getattr(row, "valuenum", None)) else None
        valueuom = str(row.valueuom) if pd.notna(getattr(row, "valueuom", None)) else None

        nr = normalizer.normalize("chartevents", itemid, label, valuenum, valueuom)
        if nr is None:
            continue

        stay_id = int(row.stay_id)
        dup_key = f"chartevents|{stay_id}|{ct.isoformat()}|{nr.canonical_name}|{itemid}"
        dup_counter[dup_key] += 1
        extra = str(dup_counter[dup_key] - 1) if dup_counter[dup_key] > 1 else ""

        uid = generate_event_uid(
            "chartevents", stay_id, ct.to_pydatetime(),
            nr.canonical_name, raw_id=str(itemid) if itemid else None,
            extra_discriminator=extra,
        )
        events.append(CanonicalEvent(
            subject_id=int(row.subject_id),
            hadm_id=int(row.hadm_id),
            stay_id=stay_id,
            event_time=ct.to_pydatetime(),
            event_uid=uid,
            source_table="chartevents",
            event_category="vital",
            canonical_name=nr.canonical_name,
            value_num=nr.normalized_value,
            unit=nr.canonical_unit,
            raw_id=str(itemid) if itemid else None,
            raw_label=nr.raw_label,
            metadata_json=_metadata_blob(
                original_value=nr.original_value,
                original_unit=valueuom,
                is_ambiguous=nr.is_ambiguous,
                codebook_version="v0.1",
            ),
        ))
    return events


def _classify_input_event(row: object) -> str:
    """Distinguish bolus from continuous infusion using order metadata.

    Implements the direct-quote anchor pattern::

        WHEN _item_class = 'INTERMITTENT' ... THEN 'intm'
        WHEN _item_class = 'CONTINUOUS'   ... THEN 'cont'
    """
    ocd = str(getattr(row, "ordercategorydescription", "") or "").lower()
    ocn = str(getattr(row, "ordercategoryname", "") or "").lower()
    if "bolus" in ocd or "bolus" in ocn:
        return "med_bolus"
    rate = getattr(row, "rate", None)
    if rate is not None and pd.notna(rate) and float(rate) > 0:
        return "med_infusion"
    return "med_bolus"


def _build_inputevents(
    df: pd.DataFrame,
    normalizer: Normalizer,
) -> list[CanonicalEvent]:
    """inputevents → med_bolus / med_infusion events with interval semantics.

    Preserves ``orderid`` / ``linkorderid`` in metadata (direct-quote anchors).
    """
    events: list[CanonicalEvent] = []
    dup_counter: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        st = _safe_ts(row.starttime)
        if st is None:
            continue
        et = _safe_ts(getattr(row, "endtime", None))

        itemid = int(row.itemid) if pd.notna(row.itemid) else None
        label = getattr(row, "label", None)

        amount = float(row.amount) if pd.notna(getattr(row, "amount", None)) else None
        amountuom = str(row.amountuom) if pd.notna(getattr(row, "amountuom", None)) else None
        rate = float(row.rate) if pd.notna(getattr(row, "rate", None)) else None
        rateuom = str(row.rateuom) if pd.notna(getattr(row, "rateuom", None)) else None

        value = amount
        unit = amountuom

        nr = normalizer.normalize("inputevents", itemid, label, value, unit)
        if nr is None:
            continue

        stay_id = int(row.stay_id)
        category = _classify_input_event(row)

        orderid = getattr(row, "orderid", None)
        linkorderid = getattr(row, "linkorderid", None)

        dup_key = f"inputevents|{stay_id}|{st.isoformat()}|{nr.canonical_name}|{itemid}|{orderid}"
        dup_counter[dup_key] += 1
        extra = str(dup_counter[dup_key] - 1) if dup_counter[dup_key] > 1 else ""

        uid = generate_event_uid(
            "inputevents", stay_id, st.to_pydatetime(),
            nr.canonical_name,
            raw_id=str(itemid) if itemid else None,
            extra_discriminator=f"{orderid or ''}|{extra}",
        )

        events.append(CanonicalEvent(
            subject_id=int(row.subject_id),
            hadm_id=int(row.hadm_id),
            stay_id=stay_id,
            event_time=st.to_pydatetime(),
            event_time_end=et.to_pydatetime() if et is not None else None,
            event_uid=uid,
            source_table="inputevents",
            event_category=category,
            canonical_name=nr.canonical_name,
            value_num=nr.normalized_value,
            unit=nr.canonical_unit,
            raw_id=str(itemid) if itemid else None,
            raw_label=nr.raw_label,
            metadata_json=_metadata_blob(
                original_value=nr.original_value,
                original_unit=amountuom,
                rate=rate,
                rate_unit=rateuom,
                order_id=orderid,
                link_order_id=linkorderid,
                is_ambiguous=nr.is_ambiguous,
                codebook_version="v0.1",
            ),
        ))
    return events


def _build_procedureevents(
    df: pd.DataFrame,
    normalizer: Normalizer,
) -> list[CanonicalEvent]:
    """procedureevents → procedure START/END events (interval semantics).

    Direct-quote anchor: ``code: [PROCEDURE, START, col(itemid)]`` /
    ``code: [PROCEDURE, END, col(itemid)]``.
    """
    events: list[CanonicalEvent] = []
    dup_counter: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        st = _safe_ts(row.starttime)
        if st is None:
            continue
        et = _safe_ts(getattr(row, "endtime", None))

        itemid = int(row.itemid) if pd.notna(row.itemid) else None
        label = getattr(row, "label", None) if hasattr(row, "label") else None

        nr = normalizer.normalize("procedureevents", itemid, label)
        if nr is None:
            continue

        stay_id = int(row.stay_id)
        orderid = getattr(row, "orderid", None)

        dup_key = f"procedureevents|{stay_id}|{st.isoformat()}|{nr.canonical_name}|{itemid}"
        dup_counter[dup_key] += 1
        extra = str(dup_counter[dup_key] - 1) if dup_counter[dup_key] > 1 else ""

        uid = generate_event_uid(
            "procedureevents", stay_id, st.to_pydatetime(),
            nr.canonical_name, raw_id=str(itemid) if itemid else None,
            extra_discriminator=extra,
        )
        events.append(CanonicalEvent(
            subject_id=int(row.subject_id),
            hadm_id=int(row.hadm_id),
            stay_id=stay_id,
            event_time=st.to_pydatetime(),
            event_time_end=et.to_pydatetime() if et is not None else None,
            event_uid=uid,
            source_table="procedureevents",
            event_category="procedure",
            canonical_name=nr.canonical_name,
            raw_id=str(itemid) if itemid else None,
            raw_label=nr.raw_label,
            metadata_json=_metadata_blob(
                order_id=orderid,
                is_ambiguous=nr.is_ambiguous,
                codebook_version="v0.1",
            ),
        ))
    return events


def _build_outputevents(
    df: pd.DataFrame,
    normalizer: Normalizer,
) -> list[CanonicalEvent]:
    """outputevents → output events (e.g. urine output)."""
    events: list[CanonicalEvent] = []
    dup_counter: Counter[str] = Counter()

    for row in df.itertuples(index=False):
        ct = _safe_ts(row.charttime)
        if ct is None:
            continue

        itemid = int(row.itemid) if pd.notna(row.itemid) else None
        label = getattr(row, "label", None) if hasattr(row, "label") else None
        value = float(row.value) if pd.notna(getattr(row, "value", None)) else None
        valueuom = str(row.valueuom) if pd.notna(getattr(row, "valueuom", None)) else None

        nr = normalizer.normalize("outputevents", itemid, label, value, valueuom)
        if nr is None:
            continue

        stay_id = int(row.stay_id)
        dup_key = f"outputevents|{stay_id}|{ct.isoformat()}|{nr.canonical_name}|{itemid}"
        dup_counter[dup_key] += 1
        extra = str(dup_counter[dup_key] - 1) if dup_counter[dup_key] > 1 else ""

        uid = generate_event_uid(
            "outputevents", stay_id, ct.to_pydatetime(),
            nr.canonical_name, raw_id=str(itemid) if itemid else None,
            extra_discriminator=extra,
        )
        events.append(CanonicalEvent(
            subject_id=int(row.subject_id),
            hadm_id=int(row.hadm_id),
            stay_id=stay_id,
            event_time=ct.to_pydatetime(),
            event_uid=uid,
            source_table="outputevents",
            event_category="output",
            canonical_name=nr.canonical_name,
            value_num=nr.normalized_value,
            unit=nr.canonical_unit,
            raw_id=str(itemid) if itemid else None,
            raw_label=nr.raw_label,
            metadata_json=_metadata_blob(
                original_value=nr.original_value,
                original_unit=valueuom,
                is_ambiguous=nr.is_ambiguous,
                codebook_version="v0.1",
            ),
        ))
    return events


# ---------------------------------------------------------------------------
# Stay lookup builder
# ---------------------------------------------------------------------------

def build_stay_lookup(icustays_df: pd.DataFrame) -> dict[tuple[int, int], int]:
    """Map (subject_id, hadm_id) → stay_id for labevents join.

    When a patient has multiple ICU stays for the same admission, the
    *first* stay is used (matching how labs are typically attributed).
    """
    lookup: dict[tuple[int, int], int] = {}
    sorted_df = icustays_df.sort_values("intime")
    for row in sorted_df.itertuples(index=False):
        key = (int(row.subject_id), int(row.hadm_id))
        if key not in lookup:
            lookup[key] = int(row.stay_id)
    return lookup


# ---------------------------------------------------------------------------
# Build stats
# ---------------------------------------------------------------------------

@dataclass
class TimelineBuildStats:
    """Counts per source table after a build run."""
    events_per_table: dict[str, int] = field(default_factory=dict)
    total_events: int = 0
    stays_built: int = 0

    def summary(self) -> str:
        lines = [
            f"Total events: {self.total_events}",
            f"Stays with events: {self.stays_built}",
        ]
        for tbl, n in sorted(self.events_per_table.items()):
            lines.append(f"  {tbl}: {n}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_stay_timeline(
    stay_id: int,
    *,
    labevents: Optional[pd.DataFrame] = None,
    chartevents: Optional[pd.DataFrame] = None,
    inputevents: Optional[pd.DataFrame] = None,
    procedureevents: Optional[pd.DataFrame] = None,
    outputevents: Optional[pd.DataFrame] = None,
    normalizer: Normalizer,
    stay_lookup: Optional[dict[tuple[int, int], int]] = None,
) -> list[CanonicalEvent]:
    """Build a sorted canonical timeline for a single ICU stay.

    Each source DataFrame should already be filtered to the target stay_id
    (or subject_id/hadm_id for labevents).
    """
    all_events: list[CanonicalEvent] = []

    if labevents is not None and not labevents.empty:
        if stay_lookup is None:
            raise ValueError("stay_lookup required for labevents (maps hadm_id→stay_id).")
        all_events.extend(_build_labevents(labevents, normalizer, stay_lookup))

    if chartevents is not None and not chartevents.empty:
        all_events.extend(_build_chartevents(chartevents, normalizer))

    if inputevents is not None and not inputevents.empty:
        all_events.extend(_build_inputevents(inputevents, normalizer))

    if procedureevents is not None and not procedureevents.empty:
        all_events.extend(_build_procedureevents(procedureevents, normalizer))

    if outputevents is not None and not outputevents.empty:
        all_events.extend(_build_outputevents(outputevents, normalizer))

    return events_sorted(all_events)


def build_all_timelines(
    settings: Settings,
    *,
    stay_ids: Optional[list[int]] = None,
    max_stays: Optional[int] = None,
    max_rows_per_table: Optional[int] = None,
) -> tuple[dict[int, list[CanonicalEvent]], TimelineBuildStats]:
    """Build canonical timelines for all (or selected) ICU stays.

    Optional max_rows_per_table limits rows read per source table (for faster
    integration tests on real data). Does not apply to icustays.

    Returns ``(timelines_dict, stats)`` where *timelines_dict* maps
    ``stay_id → sorted list[CanonicalEvent]``.
    """
    from mimic_triggerbench.data_access.tables import load_table_dataframe

    codebooks = load_all_codebooks()
    normalizer = Normalizer(codebooks)

    logger.info("Loading icustays …")
    icustays = load_table_dataframe(settings, "icustays")
    stay_lookup = build_stay_lookup(icustays)

    if stay_ids is not None:
        target_stays = [s for s in stay_ids if s in icustays["stay_id"].values]
    elif max_stays is not None:
        target_stays = icustays["stay_id"].unique()[:max_stays].tolist()
    else:
        target_stays = icustays["stay_id"].unique().tolist()

    logger.info("Building timelines for %d stays …", len(target_stays))

    source_tables = _load_source_tables(settings, max_rows_per_table=max_rows_per_table)
    stats = TimelineBuildStats()
    timelines: dict[int, list[CanonicalEvent]] = {}

    for sid in target_stays:
        icu_row = icustays[icustays["stay_id"] == sid].iloc[0]
        subject_id = int(icu_row["subject_id"])
        hadm_id = int(icu_row["hadm_id"])

        sliced = _slice_sources_for_stay(source_tables, sid, subject_id, hadm_id)
        normalizer.reset_stats()

        timeline = build_stay_timeline(
            sid,
            normalizer=normalizer,
            stay_lookup=stay_lookup,
            **sliced,
        )
        if timeline:
            timelines[sid] = timeline
            for ev in timeline:
                stats.events_per_table[ev.source_table] = (
                    stats.events_per_table.get(ev.source_table, 0) + 1
                )

    stats.total_events = sum(len(t) for t in timelines.values())
    stats.stays_built = len(timelines)
    logger.info("Timeline build complete. %s", stats.summary())
    return timelines, stats


# ---------------------------------------------------------------------------
# Helpers for bulk build
# ---------------------------------------------------------------------------

def _load_source_tables(
    settings: Settings,
    *,
    max_rows_per_table: Optional[int] = None,
) -> dict[str, pd.DataFrame]:
    from mimic_triggerbench.data_access.tables import load_table_dataframe

    tables: dict[str, pd.DataFrame] = {}
    for name in ("labevents", "chartevents", "inputevents", "procedureevents", "outputevents"):
        try:
            df = load_table_dataframe(settings, name, nrows=max_rows_per_table)
            _ensure_datetime_cols(df, name)
            tables[name] = df
            logger.info("Loaded %s: %d rows", name, len(df))
        except (FileNotFoundError, KeyError) as e:
            logger.warning("Source table %s not available; skipping: %s", name, e)
        except Exception as e:  # noqa: BLE001 - tolerate corrupted/unreadable files per table
            logger.warning("Source table %s failed to load; skipping: %s", name, e)
    return tables


_DATETIME_COLS: dict[str, list[str]] = {
    "labevents": ["charttime"],
    "chartevents": ["charttime"],
    "inputevents": ["starttime", "endtime"],
    "procedureevents": ["starttime", "endtime"],
    "outputevents": ["charttime"],
}


def _ensure_datetime_cols(df: pd.DataFrame, table: str) -> None:
    for col in _DATETIME_COLS.get(table, []):
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")


def _slice_sources_for_stay(
    source_tables: dict[str, pd.DataFrame],
    stay_id: int,
    subject_id: int,
    hadm_id: int,
) -> dict[str, pd.DataFrame]:
    """Filter loaded source tables to rows relevant for one stay."""
    sliced: dict[str, pd.DataFrame] = {}

    if "labevents" in source_tables:
        lt = source_tables["labevents"]
        sliced["labevents"] = lt[
            (lt["subject_id"] == subject_id) & (lt["hadm_id"] == hadm_id)
        ]

    for name in ("chartevents", "inputevents", "procedureevents", "outputevents"):
        if name in source_tables:
            t = source_tables[name]
            sliced[name] = t[t["stay_id"] == stay_id]

    return sliced
