"""Canonical timeline event schema and UID generation (Phase 4).

Every raw MIMIC-IV event is projected into this unified representation
before downstream labeling, replay, or agent consumption.

UID generation uses UUID-v5 with a project-specific namespace so that the
same raw row always produces the same event_uid, and rebuilds are
byte-identical.  Implements the direct-quote anchor pattern::

    uuid_generate_v5(ns_medication_administration_icu.uuid,
                     ie.stay_id || '-' || ie.orderid || '-' || ie.itemid)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Deterministic event UID
# ---------------------------------------------------------------------------

NAMESPACE_TIMELINE = uuid.UUID("7a3c9f1e-4d2b-4e8a-b5c6-1f2e3d4a5b6c")


def generate_event_uid(
    source_table: str,
    stay_id: int,
    event_time: datetime,
    canonical_name: str,
    *,
    raw_id: Optional[str] = None,
    extra_discriminator: str = "",
) -> str:
    """Create a deterministic UUID-v5 for a canonical timeline event.

    The hash input concatenates table, stay, timestamp, concept, raw id,
    and an optional discriminator to handle duplicate rows that share all
    other key fields.
    """
    ts_str = event_time.isoformat()
    raw_part = raw_id if raw_id is not None else ""
    key = f"{source_table}|{stay_id}|{ts_str}|{canonical_name}|{raw_part}|{extra_discriminator}"
    return str(uuid.uuid5(NAMESPACE_TIMELINE, key))


# ---------------------------------------------------------------------------
# Canonical event model
# ---------------------------------------------------------------------------


class CanonicalEvent(BaseModel):
    """Single row of the canonical per-stay timeline.

    Fields follow the Phase 4 spec in IMPLEMENTATION_TASKS.md.
    """

    subject_id: int
    hadm_id: int
    stay_id: int

    event_time: datetime
    event_time_end: Optional[datetime] = None

    event_uid: str = Field(
        ...,
        description="Deterministic UUID-v5 identifier for this event row.",
    )

    source_table: str
    event_category: str = Field(
        ...,
        description=(
            "High-level category: lab, vital, med_bolus, med_infusion, "
            "procedure, output, other."
        ),
    )
    canonical_name: str

    value_num: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None

    raw_id: Optional[str] = Field(
        default=None,
        description="Primary raw identifier from source table (itemid, orderid, etc.).",
    )
    raw_label: Optional[str] = None

    metadata_json: str = Field(
        default="{}",
        description="JSON blob carrying mapping provenance and extra raw fields.",
    )

    model_config = ConfigDict(frozen=True)

    # -- deterministic tie-break sort key ----------------------------------

    _CATEGORY_ORDER: dict[str, int] = {
        "lab": 0,
        "vital": 1,
        "med_bolus": 2,
        "med_infusion": 3,
        "procedure": 4,
        "output": 5,
        "other": 9,
    }

    @property
    def sort_key(self) -> tuple[datetime, int, str, str, str]:
        """Fixed tie-break key: (event_time, category_rank, source_table, canonical_name, event_uid)."""
        rank = self._CATEGORY_ORDER.get(self.event_category, 8)
        return (self.event_time, rank, self.source_table, self.canonical_name, self.event_uid)

    def metadata_dict(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)


def events_sorted(events: list[CanonicalEvent]) -> list[CanonicalEvent]:
    """Return a copy of *events* in deterministic timeline order."""
    return sorted(events, key=lambda e: e.sort_key)
