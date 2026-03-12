"""Load and validate YAML codebooks (Phase 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .codebook_models import Codebook

_MAPPINGS_DIR = Path(__file__).resolve().parent / "mappings"

_CODEBOOK_FILES: Dict[str, str] = {
    "labs": "labs.yaml",
    "vitals": "vitals.yaml",
    "meds": "meds.yaml",
    "procedures": "procedures.yaml",
}


def load_codebook(domain: str) -> Codebook:
    """Load a single codebook by domain name."""
    filename = _CODEBOOK_FILES.get(domain)
    if filename is None:
        raise FileNotFoundError(
            f"Unknown codebook domain {domain!r}. "
            f"Available: {sorted(_CODEBOOK_FILES)}"
        )
    path = _MAPPINGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Codebook file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Codebook.model_validate(raw)


def load_all_codebooks() -> Dict[str, Codebook]:
    """Load all codebooks and return a ``{domain: Codebook}`` dict."""
    return {domain: load_codebook(domain) for domain in _CODEBOOK_FILES}


def list_codebook_domains() -> List[str]:
    """Return available codebook domain names."""
    return sorted(_CODEBOOK_FILES.keys())
