from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


TaskName = Literal["hyperkalemia", "hypoglycemia", "hypotension"]
UrgencyLevel = Literal["low", "medium", "high", "critical"]


# ---------------------------------------------------------------------------
# Final benchmark output (Phase 6.5 frozen contract)
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    source_table: str
    canonical_name: str
    event_time: datetime
    value: Union[str, int, float]
    why_relevant: str

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolTraceItem(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    returned_count: int = Field(..., ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkOutput(BaseModel):
    episode_id: str
    task_name: TaskName
    trigger_detected: bool
    trigger_type: str
    decision_time: datetime
    urgency_level: UrgencyLevel
    recommended_next_steps: List[str]
    recommended_action_families: List[str]
    evidence: List[EvidenceItem]
    missing_information: List[str]
    abstain: bool
    abstain_reason: Optional[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    tool_trace: List[ToolTraceItem]

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Benchmark episode input (context fed to runners / agent)
# ---------------------------------------------------------------------------


class EpisodeInput(BaseModel):
    """Structured context a runner receives for a single benchmark episode.

    ``accepted_action_families`` carries the protocol-derived gold set.
    Rule baselines may use it directly; the agent may have it withheld.
    """

    episode_id: str
    task_name: TaskName
    stay_id: int
    hadm_id: int
    subject_id: int
    decision_time: datetime
    context_start: datetime
    mandatory_evidence_types: List[str] = Field(default_factory=list)
    accepted_action_families: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Tool result envelope (every replay tool returns this)
# ---------------------------------------------------------------------------


class TimeRange(BaseModel):
    start: datetime
    end: datetime

    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(BaseModel):
    source_tables: List[str]
    mapping_version: str = "v0.1"

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolResultRow(BaseModel):
    canonical_name: str
    event_time: datetime
    value: Optional[Union[str, int, float]] = None
    unit: Optional[str] = None
    source_table: str
    event_uid: str
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolResult(BaseModel):
    tool_name: str
    queried_time_range: TimeRange
    provenance: Provenance
    results: List[ToolResultRow]
    result_count: int = Field(..., ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Intermediate agent state (accumulated during the agent loop, Phase 9)
# ---------------------------------------------------------------------------


class IntermediateAgentState(BaseModel):
    episode_id: str
    task_name: TaskName
    decision_time: datetime
    tool_calls: List[ToolTraceItem] = Field(default_factory=list)
    gathered_evidence: List[EvidenceItem] = Field(default_factory=list)
    reasoning_notes: List[str] = Field(default_factory=list)
    iteration: int = Field(0, ge=0)

    model_config = ConfigDict(extra="forbid", frozen=True)

