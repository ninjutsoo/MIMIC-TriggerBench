from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .settings import DataBackend, Settings


def _get_path(var_name: str) -> Optional[Path]:
    v = os.getenv(var_name)
    if not v:
        return None
    return Path(v).expanduser()


def load_settings(dotenv_path: str | Path | None = ".env") -> Settings:
    """Load settings from `.env` and environment variables.

    Environment variables:
    - MIMIC_BACKEND: files | postgres
    - MIMIC_ROOT
    - MIMIC_NOTE_ROOT (optional)
    - POSTGRES_DSN (for postgres backend)
    """
    if dotenv_path is not None and Path(dotenv_path).exists():
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # still allow env vars even if .env missing
        load_dotenv()

    backend_raw = (os.getenv("MIMIC_BACKEND") or "files").strip().lower()
    backend = DataBackend(backend_raw)

    settings = Settings(
        backend=backend,
        mimic_root=_get_path("MIMIC_ROOT"),
        mimic_note_root=_get_path("MIMIC_NOTE_ROOT"),
        postgres_dsn=os.getenv("POSTGRES_DSN"),
    )

    if settings.backend == DataBackend.FILES and settings.mimic_root is None:
        raise ValueError("MIMIC_ROOT is required when MIMIC_BACKEND=files")
    if settings.backend == DataBackend.POSTGRES and not settings.postgres_dsn:
        raise ValueError("POSTGRES_DSN is required when MIMIC_BACKEND=postgres")

    return settings

