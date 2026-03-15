"""As-of time slicing and filtering for canonical timelines (Phase 4).

Every downstream consumer (replay tools, trigger detectors, label generators)
must access the timeline through these utilities to enforce the temporal
boundary contract: **no event whose ``event_time`` is after the query
timestamp may be returned**.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Sequence

from .models import CanonicalEvent


def slice_as_of(
    events: Sequence[CanonicalEvent],
    as_of: datetime,
) -> list[CanonicalEvent]:
    """Return events with ``event_time <= as_of``, preserving sort order.

    This is the primary temporal boundary enforcement point.  Future events
    are unconditionally excluded regardless of ``event_time_end``.
    """
    return [e for e in events if e.event_time <= as_of]


def slice_window(
    events: Sequence[CanonicalEvent],
    start: datetime,
    end: datetime,
) -> list[CanonicalEvent]:
    """Return events within ``[start, end]`` (inclusive on both ends)."""
    return [e for e in events if start <= e.event_time <= end]


def slice_lookback(
    events: Sequence[CanonicalEvent],
    as_of: datetime,
    hours_back: float,
) -> list[CanonicalEvent]:
    """Return events in ``[as_of - hours_back, as_of]``."""
    start = as_of - timedelta(hours=hours_back)
    return slice_window(events, start, as_of)


def filter_by_category(
    events: Sequence[CanonicalEvent],
    categories: set[str],
) -> list[CanonicalEvent]:
    """Keep only events whose ``event_category`` is in *categories*."""
    return [e for e in events if e.event_category in categories]


def filter_by_canonical_name(
    events: Sequence[CanonicalEvent],
    names: set[str],
) -> list[CanonicalEvent]:
    """Keep only events whose ``canonical_name`` is in *names*."""
    return [e for e in events if e.canonical_name in names]


def filter_by_source_table(
    events: Sequence[CanonicalEvent],
    tables: set[str],
) -> list[CanonicalEvent]:
    """Keep only events from the given source tables."""
    return [e for e in events if e.source_table in tables]


def get_latest_value(
    events: Sequence[CanonicalEvent],
    canonical_name: str,
    as_of: datetime,
) -> Optional[CanonicalEvent]:
    """Return the most recent event for *canonical_name* at or before *as_of*.

    Returns ``None`` if no matching event exists.
    """
    candidates = [
        e for e in events
        if e.canonical_name == canonical_name and e.event_time <= as_of
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.event_time)


def get_recent_values(
    events: Sequence[CanonicalEvent],
    canonical_name: str,
    as_of: datetime,
    hours_back: float,
) -> list[CanonicalEvent]:
    """Return events for *canonical_name* within ``[as_of - hours_back, as_of]``."""
    start = as_of - timedelta(hours=hours_back)
    return [
        e for e in events
        if e.canonical_name == canonical_name and start <= e.event_time <= as_of
    ]


def active_infusions_at(
    events: Sequence[CanonicalEvent],
    as_of: datetime,
) -> list[CanonicalEvent]:
    """Return infusion events that are *active* at *as_of*.

    An infusion is active when ``event_time <= as_of`` and either
    ``event_time_end`` is ``None`` or ``event_time_end > as_of``.
    Only ``med_infusion`` category events are considered.
    """
    return [
        e for e in events
        if (
            e.event_category == "med_infusion"
            and e.event_time <= as_of
            and (e.event_time_end is None or e.event_time_end > as_of)
        )
    ]
