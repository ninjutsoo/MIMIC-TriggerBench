"""Pydantic models for normalization codebooks (Phase 3)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class UnitConversion(BaseModel):
    """Rule to convert a value from one unit to the codebook's canonical unit."""

    from_unit: str = Field(..., min_length=1)
    factor: float = Field(..., description="Multiply raw value by this factor.")
    offset: float = Field(default=0.0, description="Add this after multiplication.")

    model_config = ConfigDict(frozen=True)


class CodebookEntry(BaseModel):
    """A single canonical concept with its raw-to-canonical mapping rules."""

    canonical_name: str = Field(..., min_length=1)
    description: str = ""
    source_tables: List[str] = Field(..., min_length=1)
    raw_itemids: List[int] = Field(default_factory=list)
    raw_labels: List[str] = Field(default_factory=list)
    canonical_unit: Optional[str] = None
    conversions: List[UnitConversion] = Field(default_factory=list)
    ambiguity_notes: List[str] = Field(default_factory=list)
    is_ambiguous: bool = False

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def _must_have_identifiers(self) -> CodebookEntry:
        if not self.raw_itemids and not self.raw_labels:
            raise ValueError(
                f"Entry '{self.canonical_name}': supply at least one "
                "raw_itemid or raw_label."
            )
        return self


class Codebook(BaseModel):
    """A versioned collection of canonical concept mappings for one domain."""

    name: str = Field(..., min_length=1)
    version: str = Field(..., pattern=r"^v\d+\.\d+$")
    description: str = ""
    entries: List[CodebookEntry] = Field(..., min_length=1)

    model_config = ConfigDict(frozen=True)
