from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from mimic_triggerbench.mimic_tables import iter_table_specs, resolve_table_path

from .settings import DataBackend, Settings


def _get_path(var_name: str) -> Optional[Path]:
    v = os.getenv(var_name)
    if not v:
        return None
    return Path(v).expanduser()


def _has_required_files(mimic_root: Path) -> bool:
    return all(resolve_table_path(mimic_root, spec.table_name) is not None for spec in iter_table_specs())


def _discover_repo_mimic_root() -> Path | None:
    """Discover a repo-local MIMIC root when present.

    Expected local layout (mirrors PhysioNet download structure), with
    table files available as .parquet, .csv.gz, or .csv:
    - <repo>/physionet.org/files/mimiciv/<version>/{icu,hosp}/*
    """
    # config/ is <repo>/src/mimic_triggerbench/config
    repo_root = Path(__file__).resolve().parents[3]
    base = repo_root / "physionet.org" / "files" / "mimiciv"
    if not base.exists():
        return None

    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None

    for version_dir in sorted(candidates, key=lambda p: p.name, reverse=True):
        if _has_required_files(version_dir):
            return version_dir
    return None


def load_settings(dotenv_path: str | Path | None = ".env") -> Settings:
    """Load settings from `.env` and environment variables.

    Environment variables:
    - MIMIC_BACKEND: files | postgres
    - MIMIC_ROOT
    - MIMIC_NOTE_ROOT (optional)
    - POSTGRES_DSN (for postgres backend)
    - HF_HOME (optional): Hugging Face cache root (e.g. ``/data/Amin``); read by ``huggingface_hub``.
    - TRIGGERBENCH_HF_MODEL_ID (optional): HF repo id for local LLM/RAG baselines.
    - TRIGGERBENCH_CUDA_DEVICE (optional): e.g. ``cuda:0`` for single-GPU runs.
    - TRIGGERBENCH_RUN_HF_TESTS: set to ``1`` to run GPU HF baseline tests and real-data LLM/RAG sections.
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
        discovered = _discover_repo_mimic_root()
        if discovered is not None:
            settings = Settings(
                backend=settings.backend,
                mimic_root=discovered,
                mimic_note_root=settings.mimic_note_root,
                postgres_dsn=settings.postgres_dsn,
            )
        else:
            raise ValueError(
                "MIMIC_ROOT is required when MIMIC_BACKEND=files "
                "(and no repo-local physionet.org mirror was discovered)."
            )
    if settings.backend == DataBackend.POSTGRES and not settings.postgres_dsn:
        raise ValueError("POSTGRES_DSN is required when MIMIC_BACKEND=postgres")

    return settings
