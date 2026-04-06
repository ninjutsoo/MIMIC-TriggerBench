"""Pydantic models for benchmark episodes and deterministic episode ID generation (Phase 5).

An episode is a decision point where the agent must act (positive) or should
abstain (negative).  The model strictly separates:

1. **trigger_label** -- whether the clinical trigger fired at ``decision_time``
2. **accepted_action_families** -- protocol-derived gold actions (what *should* be done)
3. **observed_action_families** -- what clinicians *actually* did in the window
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

NAMESPACE_EPISODE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def generate_episode_id(
    task_name: str,
    stay_id: int,
    decision_time: datetime,
    trigger_type: str,
    *,
    extra: str = "",
) -> str:
    """Deterministic UUID-v5 for an episode row."""
    key = f"{task_name}|{stay_id}|{decision_time.isoformat()}|{trigger_type}|{extra}"
    return str(uuid.uuid5(NAMESPACE_EPISODE, key))


class Episode(BaseModel):
    """A single benchmark episode (Phase 5 required schema)."""

    episode_id: str
    task_name: str
    subject_id: int
    hadm_id: int
    stay_id: int

    decision_time: datetime
    context_start: datetime = Field(
        ...,
        description="Earliest event time the agent is allowed to see for this episode.",
    )

    trigger_label: bool = Field(
        ..., description="True if trigger condition is met at decision_time."
    )
    trigger_type: Literal["positive", "negative"]
    trigger_value: Optional[float] = Field(
        default=None, description="The signal value that caused (or did not cause) the trigger."
    )

    accepted_action_families: List[str] = Field(
        default_factory=list,
        description="Protocol-derived gold action families (what the protocol says to do).",
    )
    observed_action_families: List[str] = Field(
        default_factory=list,
        description="Action families actually observed in clinician records within the action window.",
    )
    mandatory_evidence_types: List[str] = Field(
        default_factory=list,
        description="Evidence types the agent must retrieve (from task spec).",
    )

    split: Optional[str] = Field(
        default=None,
        description="train / val / test -- assigned in Phase 6.",
    )

    model_config = ConfigDict(frozen=True)
