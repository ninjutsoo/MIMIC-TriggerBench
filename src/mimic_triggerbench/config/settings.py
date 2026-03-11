from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic import ConfigDict


class DataBackend(str, Enum):
    FILES = "files"
    POSTGRES = "postgres"


class Settings(BaseModel):
    """Minimal config model for Phase 0/1.

    This will be extended in later phases; for now it only captures enough
    information to drive a data inventory command.
    """

    backend: DataBackend = Field(
        default=DataBackend.FILES,
        description="Where raw MIMIC data is loaded from.",
    )
    mimic_root: Optional[Path] = Field(
        default=None,
        description="Root directory containing MIMIC-IV files (e.g. /data/mimiciv).",
    )
    mimic_note_root: Optional[Path] = Field(
        default=None,
        description="Root directory containing MIMIC-IV-Note files.",
    )
    postgres_dsn: Optional[str] = Field(
        default=None,
        description="SQLAlchemy-style DSN for PostgreSQL, if using a database backend.",
    )

    model_config = ConfigDict(frozen=True)

