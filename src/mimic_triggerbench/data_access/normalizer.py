"""Event normalization pipeline (Phase 3).

Maps raw MIMIC-IV event data (itemids, labels, values, units) to canonical
concepts using loaded codebooks.  Tracks unmapped and ambiguous rows so
high-frequency gaps are surfaced rather than silently ignored.
"""

from __future__ import annotations

import fnmatch
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .codebook_models import Codebook, CodebookEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NormalizationResult:
    """Result of normalizing a single raw event row."""

    canonical_name: str
    original_value: Optional[float]
    normalized_value: Optional[float]
    canonical_unit: Optional[str]
    source_table: str
    raw_itemid: Optional[int]
    raw_label: Optional[str]
    is_ambiguous: bool
    ambiguity_notes: tuple[str, ...]


@dataclass
class NormalizationStats:
    """Accumulated statistics from normalization runs."""

    mapped: int = 0
    unmapped: int = 0
    ambiguous: int = 0
    unmapped_itemids: Counter = field(default_factory=Counter)
    unmapped_labels: Counter = field(default_factory=Counter)

    def summary(self) -> Dict[str, object]:
        return {
            "mapped": self.mapped,
            "unmapped": self.unmapped,
            "ambiguous": self.ambiguous,
            "top_unmapped_itemids": self.unmapped_itemids.most_common(20),
            "top_unmapped_labels": self.unmapped_labels.most_common(20),
        }


class Normalizer:
    """Maps raw MIMIC events to canonical concepts via codebook lookup.

    Lookup priority: itemid first, then label pattern (fnmatch, case-insensitive).
    """

    def __init__(self, codebooks: Dict[str, Codebook]) -> None:
        self._codebooks = codebooks
        self._itemid_index: Dict[int, CodebookEntry] = {}
        self._label_entries: List[tuple[str, CodebookEntry]] = []
        self.stats = NormalizationStats()
        self._build_indices()

    def _build_indices(self) -> None:
        for cb in self._codebooks.values():
            for entry in cb.entries:
                for iid in entry.raw_itemids:
                    self._itemid_index[iid] = entry
                for label_pattern in entry.raw_labels:
                    self._label_entries.append((label_pattern, entry))

    def _match_by_itemid(self, itemid: int) -> Optional[CodebookEntry]:
        return self._itemid_index.get(itemid)

    def _match_by_label(self, label: str) -> Optional[CodebookEntry]:
        label_lower = label.lower()
        for pattern, entry in self._label_entries:
            if fnmatch.fnmatch(label_lower, pattern.lower()):
                return entry
        return None

    def _convert_value(
        self, entry: CodebookEntry, value: Optional[float], unit: Optional[str]
    ) -> Optional[float]:
        if value is None or unit is None:
            return value
        for conv in entry.conversions:
            if conv.from_unit.lower() == unit.lower():
                return value * conv.factor + conv.offset
        return value

    def normalize(
        self,
        source_table: str,
        itemid: Optional[int],
        label: Optional[str],
        value: Optional[float] = None,
        unit: Optional[str] = None,
    ) -> Optional[NormalizationResult]:
        """Normalize a raw event row.

        Returns a ``NormalizationResult`` on match, or ``None`` if unmapped.
        Statistics are updated in either case.
        """
        entry: Optional[CodebookEntry] = None

        if itemid is not None:
            entry = self._match_by_itemid(itemid)
        if entry is None and label is not None:
            entry = self._match_by_label(label)

        if entry is None:
            self.stats.unmapped += 1
            if itemid is not None:
                self.stats.unmapped_itemids[itemid] += 1
            if label is not None:
                self.stats.unmapped_labels[label] += 1
            return None

        normalized_value = self._convert_value(entry, value, unit)
        canonical_unit = entry.canonical_unit if entry.canonical_unit else unit

        if entry.is_ambiguous:
            self.stats.ambiguous += 1
        self.stats.mapped += 1

        return NormalizationResult(
            canonical_name=entry.canonical_name,
            original_value=value,
            normalized_value=normalized_value,
            canonical_unit=canonical_unit,
            source_table=source_table,
            raw_itemid=itemid,
            raw_label=label,
            is_ambiguous=entry.is_ambiguous,
            ambiguity_notes=tuple(entry.ambiguity_notes),
        )

    def reset_stats(self) -> None:
        """Clear accumulated normalization statistics."""
        self.stats = NormalizationStats()

    def get_unmapped_report(self) -> str:
        """Generate a human-readable report of unmapped high-frequency items."""
        lines = ["# Unmapped Items Report", ""]
        s = self.stats
        lines.append(f"Total mapped: {s.mapped}")
        lines.append(f"Total unmapped: {s.unmapped}")
        lines.append(f"Total ambiguous: {s.ambiguous}")
        lines.append("")

        if s.unmapped_itemids:
            lines.append("## Top Unmapped Item IDs")
            for iid, count in s.unmapped_itemids.most_common(20):
                lines.append(f"- itemid={iid}: {count} occurrences")
            lines.append("")

        if s.unmapped_labels:
            lines.append("## Top Unmapped Labels")
            for lbl, count in s.unmapped_labels.most_common(20):
                lines.append(f"- \"{lbl}\": {count} occurrences")
            lines.append("")

        return "\n".join(lines)
