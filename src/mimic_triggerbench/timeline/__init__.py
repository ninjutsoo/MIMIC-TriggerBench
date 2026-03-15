"""Canonical timeline construction (Phase 4).

Provides:
- ``CanonicalEvent`` model and deterministic sort utilities
- ``generate_event_uid`` for reproducible UUID-v5 identifiers
- ``build_stay_timeline`` / ``build_all_timelines`` for constructing timelines
- As-of time slicing and filtering helpers
- Parquet I/O for persisted timelines
"""

from .models import CanonicalEvent, events_sorted, generate_event_uid, NAMESPACE_TIMELINE
from .builder import (
    build_stay_timeline,
    build_all_timelines,
    build_stay_lookup,
    TimelineBuildStats,
)
from .slicing import (
    slice_as_of,
    slice_window,
    slice_lookback,
    filter_by_category,
    filter_by_canonical_name,
    filter_by_source_table,
    get_latest_value,
    get_recent_values,
    active_infusions_at,
)
from .io import (
    events_to_dataframe,
    dataframe_to_events,
    write_timeline_parquet,
    read_timeline_parquet,
    TIMELINE_ARROW_SCHEMA,
)

__all__ = [
    "CanonicalEvent",
    "events_sorted",
    "generate_event_uid",
    "NAMESPACE_TIMELINE",
    "build_stay_timeline",
    "build_all_timelines",
    "build_stay_lookup",
    "TimelineBuildStats",
    "slice_as_of",
    "slice_window",
    "slice_lookback",
    "filter_by_category",
    "filter_by_canonical_name",
    "filter_by_source_table",
    "get_latest_value",
    "get_recent_values",
    "active_infusions_at",
    "events_to_dataframe",
    "dataframe_to_events",
    "write_timeline_parquet",
    "read_timeline_parquet",
    "TIMELINE_ARROW_SCHEMA",
]
