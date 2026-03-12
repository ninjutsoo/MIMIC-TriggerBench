"""Pydantic models for the Phase 3 mapping ledger contract."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MappingDecision(str, Enum):
    MAP_AS_IS = "map_as_is"
    MAP_CONVERT_UNIT = "map_convert_unit"
    NO_MAPPING = "no_mapping"
    UNSURE = "unsure"


class ReviewStatus(str, Enum):
    VALIDATED = "validated"
    PENDING = "pending"
    NEEDS_REVIEW = "needs_review"


class MappingLedgerRow(BaseModel):
    """Single row in the auditable raw->canonical mapping ledger."""

    source_table: str = Field(..., min_length=1)
    source_identifier: str = Field(..., min_length=1)
    source_label: str = Field(..., min_length=1)
    mapping_decision: MappingDecision
    canonical_concept: Optional[str] = None
    source_unit: Optional[str] = None
    target_unit: Optional[str] = None
    conversion_factor: Optional[float] = None
    conversion_offset: Optional[float] = None
    loinc_code: Optional[str] = None
    rxnorm_code: Optional[str] = None
    snomed_code: Optional[str] = None
    source_frequency_count: int = Field(..., ge=0)
    unit_frequency_summary: str = ""
    review_status: ReviewStatus
    review_note: str = ""

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _validate_decision_contract(self) -> "MappingLedgerRow":
        mapped = self.mapping_decision in {
            MappingDecision.MAP_AS_IS,
            MappingDecision.MAP_CONVERT_UNIT,
        }
        if mapped and not self.canonical_concept:
            raise ValueError("Mapped rows require canonical_concept.")
        if not mapped and self.canonical_concept is not None:
            raise ValueError("no_mapping/unsure rows must not set canonical_concept.")

        if self.mapping_decision == MappingDecision.MAP_CONVERT_UNIT:
            if not self.source_unit or not self.target_unit:
                raise ValueError("map_convert_unit rows require source_unit and target_unit.")
            if self.conversion_factor is None:
                raise ValueError("map_convert_unit rows require conversion_factor.")
            if self.conversion_offset is None:
                raise ValueError("map_convert_unit rows require conversion_offset.")
        return self

