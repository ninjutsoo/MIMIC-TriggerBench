"""Structured, timestamp-bounded replay tools (Phase 7).

Each tool operates on a canonical timeline (list of CanonicalEvent) and
returns a validated :class:`ToolResult` envelope.  The ``decision_time``
parameter acts as a hard temporal ceiling: no event with
``event_time > decision_time`` is ever returned.

Tool functions share the signature::

    def tool_fn(events, *, decision_time, **kwargs) -> ToolResult

The ``before_time`` argument supplied by the caller is always clamped to
``min(before_time, decision_time)`` so that even an erroneous or
adversarial request cannot access future data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Sequence, Union

from mimic_triggerbench.schemas.outputs import (
    Provenance,
    TimeRange,
    ToolResult,
    ToolResultRow,
)
from mimic_triggerbench.timeline.models import CanonicalEvent
from mimic_triggerbench.timeline.slicing import (
    active_infusions_at,
    filter_by_canonical_name,
    filter_by_category,
    slice_as_of,
    slice_lookback,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAPPING_VERSION = "v0.1"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp_before_time(before_time: datetime, decision_time: datetime) -> datetime:
    """Return the earlier of *before_time* and *decision_time*."""
    if before_time > decision_time:
        logger.warning(
            "before_time %s exceeds decision_time %s -- clamping",
            before_time.isoformat(),
            decision_time.isoformat(),
        )
        return decision_time
    return before_time


def _event_to_row(e: CanonicalEvent) -> ToolResultRow:
    return ToolResultRow(
        canonical_name=e.canonical_name,
        event_time=e.event_time,
        value=e.value_num if e.value_num is not None else e.value_text,
        unit=e.unit,
        source_table=e.source_table,
        event_uid=e.event_uid,
        metadata=e.metadata_dict() if e.metadata_json != "{}" else None,
    )


def _source_tables(events: Sequence[CanonicalEvent]) -> list[str]:
    return sorted({e.source_table for e in events})


def _build_result(
    tool_name: str,
    rows: list[ToolResultRow],
    time_start: datetime,
    time_end: datetime,
    source_tables: list[str],
) -> ToolResult:
    return ToolResult(
        tool_name=tool_name,
        queried_time_range=TimeRange(start=time_start, end=time_end),
        provenance=Provenance(
            source_tables=source_tables,
            mapping_version=_DEFAULT_MAPPING_VERSION,
        ),
        results=rows,
        result_count=len(rows),
    )


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def get_recent_labs(
    events: Sequence[CanonicalEvent],
    *,
    decision_time: datetime,
    before_time: datetime,
    lab_names: List[str],
    hours_back: float = 24.0,
) -> ToolResult:
    bt = _clamp_before_time(before_time, decision_time)
    start = bt - timedelta(hours=hours_back)
    sliced = slice_lookback(list(events), bt, hours_back)
    filtered = filter_by_category(sliced, {"lab"})
    if lab_names:
        filtered = filter_by_canonical_name(filtered, set(lab_names))
    rows = [_event_to_row(e) for e in filtered]
    return _build_result("get_recent_labs", rows, start, bt, _source_tables(filtered))


def get_recent_vitals(
    events: Sequence[CanonicalEvent],
    *,
    decision_time: datetime,
    before_time: datetime,
    vital_names: List[str],
    minutes_back: float = 60.0,
) -> ToolResult:
    bt = _clamp_before_time(before_time, decision_time)
    hours_back = minutes_back / 60.0
    start = bt - timedelta(hours=hours_back)
    sliced = slice_lookback(list(events), bt, hours_back)
    filtered = filter_by_category(sliced, {"vital"})
    if vital_names:
        filtered = filter_by_canonical_name(filtered, set(vital_names))
    rows = [_event_to_row(e) for e in filtered]
    return _build_result("get_recent_vitals", rows, start, bt, _source_tables(filtered))


def get_recent_meds(
    events: Sequence[CanonicalEvent],
    *,
    decision_time: datetime,
    before_time: datetime,
    med_classes: List[str],
    hours_back: float = 6.0,
) -> ToolResult:
    bt = _clamp_before_time(before_time, decision_time)
    start = bt - timedelta(hours=hours_back)
    sliced = slice_lookback(list(events), bt, hours_back)
    filtered = filter_by_category(sliced, {"med_bolus", "med_infusion"})
    if med_classes:
        filtered = filter_by_canonical_name(filtered, set(med_classes))
    rows = [_event_to_row(e) for e in filtered]
    return _build_result("get_recent_meds", rows, start, bt, _source_tables(filtered))


def get_recent_fluids(
    events: Sequence[CanonicalEvent],
    *,
    decision_time: datetime,
    before_time: datetime,
    hours_back: float = 2.0,
) -> ToolResult:
    bt = _clamp_before_time(before_time, decision_time)
    start = bt - timedelta(hours=hours_back)
    sliced = slice_lookback(list(events), bt, hours_back)
    filtered = filter_by_category(sliced, {"med_bolus", "med_infusion"})
    fluid_names = {
        "crystalloid_bolus", "normal_saline", "lactated_ringers",
        "albumin", "colloid",
    }
    filtered = [e for e in filtered if e.canonical_name in fluid_names]
    rows = [_event_to_row(e) for e in filtered]
    return _build_result("get_recent_fluids", rows, start, bt, _source_tables(filtered))


def get_recent_procedures(
    events: Sequence[CanonicalEvent],
    *,
    decision_time: datetime,
    before_time: datetime,
    hours_back: float = 12.0,
) -> ToolResult:
    bt = _clamp_before_time(before_time, decision_time)
    start = bt - timedelta(hours=hours_back)
    sliced = slice_lookback(list(events), bt, hours_back)
    filtered = filter_by_category(sliced, {"procedure"})
    rows = [_event_to_row(e) for e in filtered]
    return _build_result("get_recent_procedures", rows, start, bt, _source_tables(filtered))


def get_active_infusions(
    events: Sequence[CanonicalEvent],
    *,
    decision_time: datetime,
    before_time: datetime,
) -> ToolResult:
    bt = _clamp_before_time(before_time, decision_time)
    active = active_infusions_at(list(events), bt)
    rows = [_event_to_row(e) for e in active]
    return _build_result("get_active_infusions", rows, bt, bt, _source_tables(active))


AggregationType = Literal["raw", "mean", "min", "max"]


def get_trend(
    events: Sequence[CanonicalEvent],
    *,
    decision_time: datetime,
    before_time: datetime,
    signal_name: str,
    lookback_hours: float = 6.0,
    aggregation: AggregationType = "raw",
) -> ToolResult:
    bt = _clamp_before_time(before_time, decision_time)
    start = bt - timedelta(hours=lookback_hours)
    sliced = slice_lookback(list(events), bt, lookback_hours)
    filtered = filter_by_canonical_name(sliced, {signal_name})
    rows = [_event_to_row(e) for e in filtered]

    if aggregation != "raw" and rows:
        values = [r.value for r in rows if isinstance(r.value, (int, float))]
        if values:
            if aggregation == "mean":
                agg_val = sum(values) / len(values)
            elif aggregation == "min":
                agg_val = min(values)
            else:
                agg_val = max(values)
            agg_row = ToolResultRow(
                canonical_name=signal_name,
                event_time=bt,
                value=round(agg_val, 4),
                unit=rows[0].unit if rows else None,
                source_table="aggregation",
                event_uid=f"agg_{aggregation}",
                metadata={"aggregation": aggregation, "n_values": len(values)},
            )
            rows = [agg_row]

    return _build_result("get_trend", rows, start, bt, _source_tables(filtered))


# ---------------------------------------------------------------------------
# Tool registry (name -> callable)
# ---------------------------------------------------------------------------

TOOL_REGISTRY: Dict[str, object] = {
    "get_recent_labs": get_recent_labs,
    "get_recent_vitals": get_recent_vitals,
    "get_recent_meds": get_recent_meds,
    "get_recent_fluids": get_recent_fluids,
    "get_recent_procedures": get_recent_procedures,
    "get_active_infusions": get_active_infusions,
    "get_trend": get_trend,
}
