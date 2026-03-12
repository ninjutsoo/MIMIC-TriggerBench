"""
Data access layer for local offline MIMIC-IV and optional MIMIC-IV-Note data.

Phase 1 exposes:
- generate_inventory_report(...) – presence of required raw tables
- load_table_dataframe(...) – minimal helper to load a raw table as pandas

Phase 3 adds:
- Codebook / CodebookEntry / UnitConversion models
- load_codebook / load_all_codebooks helpers
- Normalizer with NormalizationResult / NormalizationStats
- Normalization coverage audit utilities
"""

from .inventory import generate_inventory_report
from .tables import load_table_dataframe

from .codebook_models import Codebook, CodebookEntry, UnitConversion
from .codebook_loader import load_codebook, load_all_codebooks, list_codebook_domains
from .normalizer import Normalizer, NormalizationResult, NormalizationStats
from .normalization_audit import (
    TableScanResult,
    scan_normalization_coverage,
    write_normalization_coverage_report,
)

__all__ = [
    "generate_inventory_report",
    "load_table_dataframe",
    "Codebook",
    "CodebookEntry",
    "UnitConversion",
    "load_codebook",
    "load_all_codebooks",
    "list_codebook_domains",
    "Normalizer",
    "NormalizationResult",
    "NormalizationStats",
    "TableScanResult",
    "scan_normalization_coverage",
    "write_normalization_coverage_report",
]
