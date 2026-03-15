"""Parquet I/O for canonical timelines (Phase 4).

Saves and loads timelines as partitioned Parquet files for efficient
downstream access by stay_id or task-relevant cohort.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .models import CanonicalEvent

logger = logging.getLogger(__name__)

# Explicit Arrow schema so Parquet metadata is stable across runs.
TIMELINE_ARROW_SCHEMA = pa.schema([
    ("subject_id", pa.int64()),
    ("hadm_id", pa.int64()),
    ("stay_id", pa.int64()),
    ("event_time", pa.timestamp("us")),
    ("event_time_end", pa.timestamp("us")),
    ("event_uid", pa.string()),
    ("source_table", pa.string()),
    ("event_category", pa.string()),
    ("canonical_name", pa.string()),
    ("value_num", pa.float64()),
    ("value_text", pa.string()),
    ("unit", pa.string()),
    ("raw_id", pa.string()),
    ("raw_label", pa.string()),
    ("metadata_json", pa.string()),
])


def events_to_dataframe(events: list[CanonicalEvent]) -> pd.DataFrame:
    """Convert a list of CanonicalEvent to a flat DataFrame."""
    if not events:
        return pd.DataFrame(columns=[f.name for f in TIMELINE_ARROW_SCHEMA])

    records = [
        {
            "subject_id": e.subject_id,
            "hadm_id": e.hadm_id,
            "stay_id": e.stay_id,
            "event_time": e.event_time,
            "event_time_end": e.event_time_end,
            "event_uid": e.event_uid,
            "source_table": e.source_table,
            "event_category": e.event_category,
            "canonical_name": e.canonical_name,
            "value_num": e.value_num,
            "value_text": e.value_text,
            "unit": e.unit,
            "raw_id": e.raw_id,
            "raw_label": e.raw_label,
            "metadata_json": e.metadata_json,
        }
        for e in events
    ]
    return pd.DataFrame.from_records(records)


def dataframe_to_events(df: pd.DataFrame) -> list[CanonicalEvent]:
    """Reconstruct a list of CanonicalEvent from a DataFrame."""
    events: list[CanonicalEvent] = []
    for row in df.itertuples(index=False):
        et_end = getattr(row, "event_time_end", None)
        if et_end is not None and pd.isna(et_end):
            et_end = None
        elif et_end is not None:
            et_end = pd.Timestamp(et_end).to_pydatetime()

        events.append(CanonicalEvent(
            subject_id=int(row.subject_id),
            hadm_id=int(row.hadm_id),
            stay_id=int(row.stay_id),
            event_time=pd.Timestamp(row.event_time).to_pydatetime(),
            event_time_end=et_end,
            event_uid=str(row.event_uid),
            source_table=str(row.source_table),
            event_category=str(row.event_category),
            canonical_name=str(row.canonical_name),
            value_num=float(row.value_num) if pd.notna(getattr(row, "value_num", None)) else None,
            value_text=str(row.value_text) if pd.notna(getattr(row, "value_text", None)) else None,
            unit=str(row.unit) if pd.notna(getattr(row, "unit", None)) else None,
            raw_id=str(row.raw_id) if pd.notna(getattr(row, "raw_id", None)) else None,
            raw_label=str(row.raw_label) if pd.notna(getattr(row, "raw_label", None)) else None,
            metadata_json=str(row.metadata_json) if pd.notna(getattr(row, "metadata_json", None)) else "{}",
        ))
    return events


def write_timeline_parquet(
    timelines: dict[int, list[CanonicalEvent]],
    output_dir: Path,
    *,
    partition_cols: Optional[list[str]] = None,
) -> Path:
    """Write all timelines to a single (optionally partitioned) Parquet dataset.

    Returns the path to the written dataset root.
    """
    all_events: list[CanonicalEvent] = []
    for stay_events in timelines.values():
        all_events.extend(stay_events)

    df = events_to_dataframe(all_events)
    output_dir.mkdir(parents=True, exist_ok=True)

    if partition_cols:
        table = pa.Table.from_pandas(df, schema=TIMELINE_ARROW_SCHEMA, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(output_dir),
            partition_cols=partition_cols,
        )
        logger.info(
            "Wrote partitioned timeline dataset to %s (%d events, partitioned by %s)",
            output_dir, len(df), partition_cols,
        )
    else:
        out_file = output_dir / "canonical_timeline.parquet"
        table = pa.Table.from_pandas(df, schema=TIMELINE_ARROW_SCHEMA, preserve_index=False)
        pq.write_table(table, str(out_file))
        logger.info("Wrote timeline to %s (%d events)", out_file, len(df))

    return output_dir


def read_timeline_parquet(
    path: Path,
    *,
    stay_ids: Optional[list[int]] = None,
) -> dict[int, list[CanonicalEvent]]:
    """Read canonical timelines from a Parquet file or partitioned dataset.

    Optionally filter to specific ``stay_ids`` for efficiency.
    """
    if path.is_file():
        df = pd.read_parquet(path)
    elif path.is_dir():
        df = pd.read_parquet(path)
    else:
        raise FileNotFoundError(f"Timeline path not found: {path}")

    if stay_ids is not None:
        df = df[df["stay_id"].isin(stay_ids)]

    timelines: dict[int, list[CanonicalEvent]] = {}
    for sid, group in df.groupby("stay_id", observed=True):
        group_sorted = group.sort_values(
            ["event_time", "event_category", "source_table", "canonical_name", "event_uid"]
        )
        timelines[int(sid)] = dataframe_to_events(group_sorted)
    return timelines
