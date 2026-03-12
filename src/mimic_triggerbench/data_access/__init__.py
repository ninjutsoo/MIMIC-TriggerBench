"""
Data access layer for local offline MIMIC-IV and optional MIMIC-IV-Note data.

Phase 1 exposes:
- generate_inventory_report(...) – presence of required raw tables
- load_table_dataframe(...) – minimal helper to load a raw table as pandas

Phase 3 adds:
- Codebook / CodebookEntry / UnitConversion models
- load_codebook / load_all_codebooks helpers
- Normalizer with NormalizationResult / NormalizationStats
- Mapping ledger models + loader + reconciliation checks
- Normalization coverage audit utilities
"""

from .inventory import generate_inventory_report
from .tables import load_table_dataframe

from .codebook_models import Codebook, CodebookEntry, UnitConversion
from .codebook_loader import load_codebook, load_all_codebooks, list_codebook_domains
from .normalizer import Normalizer, NormalizationResult, NormalizationStats
from .mapping_ledger_models import MappingDecision, MappingLedgerRow, ReviewStatus
from .mapping_ledger_loader import (
    MappingLedgerReconciliation,
    assert_mapping_ledger_consistent,
    load_mapping_ledger,
    mapping_ledger_path,
    reconcile_mapping_ledger,
)
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
    "MappingDecision",
    "ReviewStatus",
    "MappingLedgerRow",
    "mapping_ledger_path",
    "load_mapping_ledger",
    "reconcile_mapping_ledger",
    "assert_mapping_ledger_consistent",
    "MappingLedgerReconciliation",
    "Normalizer",
    "NormalizationResult",
    "NormalizationStats",
    "TableScanResult",
    "scan_normalization_coverage",
    "write_normalization_coverage_report",
]
