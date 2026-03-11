"""Pydantic models for versioned clinical task specifications (Phase 2)."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TriggerOperator(str, Enum):
    GTE = ">="
    LTE = "<="
    GT = ">"
    LT = "<"
    EQ = "=="


class TriggerDef(BaseModel):
    """Defines the deterministic trigger condition for a task."""

    signal: str = Field(..., description="Canonical signal name (e.g. 'potassium', 'glucose', 'map').")
    operator: TriggerOperator
    threshold: float
    unit: str
    sustained_minutes: Optional[int] = Field(
        default=None,
        description="If set, condition must hold for this many consecutive minutes.",
    )
    description: str = Field(..., description="Human-readable trigger description with clinical citation.")

    model_config = ConfigDict(frozen=True)


class ExclusionRule(BaseModel):
    """A named exclusion criterion that removes patients/episodes from the benchmark."""

    name: str
    description: str
    how_to_identify: str = Field(
        ...,
        description="How to operationalise this exclusion in MIMIC (ICD codes, flags, etc.).",
    )

    model_config = ConfigDict(frozen=True)


class EpisodeClusteringRule(BaseModel):
    """Controls how recurrent triggers within the same stay are grouped."""

    washout_minutes: int = Field(
        ..., gt=0,
        description="Minutes of the trigger condition being FALSE before a new episode can fire.",
    )
    description: str

    model_config = ConfigDict(frozen=True)


class EvidenceType(BaseModel):
    """A type of supporting evidence the agent should retrieve for context."""

    name: str
    lookback_hours: float = Field(..., gt=0)
    required: bool = Field(default=True)
    description: str = ""

    model_config = ConfigDict(frozen=True)


class ActionFamily(BaseModel):
    """A concrete downstream action family used for labels."""

    name: str
    description: str
    concrete_examples: List[str] = Field(default_factory=list)
    is_primary: bool = Field(
        default=True,
        description="Primary actions count as definitive treatment; secondary (e.g. repeat lab) do not suffice alone.",
    )

    model_config = ConfigDict(frozen=True)


class NegativeWindowDef(BaseModel):
    """Defines how negative (no-trigger) episodes are constructed."""

    trigger_false_prior_hours: float = Field(
        ..., gt=0,
        description="Trigger must have been false for this many hours before the candidate time.",
    )
    trigger_false_subsequent_hours: float = Field(
        ..., gt=0,
        description="Trigger must remain false for this many hours after (avoids near-miss leakage).",
    )
    description: str = ""

    model_config = ConfigDict(frozen=True)


class TaskSpec(BaseModel):
    """Top-level versioned task specification for a benchmark trigger-response task."""

    spec_version: str = Field(..., pattern=r"^v\d+\.\d+$")
    task_name: str
    display_name: str
    description: str

    trigger: TriggerDef
    exclusions: List[ExclusionRule] = Field(default_factory=list)
    clustering: Optional[EpisodeClusteringRule] = None

    action_window_hours: float = Field(..., gt=0)
    action_families: List[ActionFamily] = Field(..., min_length=1)

    evidence_types: List[EvidenceType] = Field(default_factory=list)
    negative_window: NegativeWindowDef

    model_config = ConfigDict(frozen=True)

    @field_validator("action_families")
    @classmethod
    def at_least_one_primary(cls, v: List[ActionFamily]) -> List[ActionFamily]:
        if not any(a.is_primary for a in v):
            raise ValueError("At least one action family must be marked as primary.")
        return v
