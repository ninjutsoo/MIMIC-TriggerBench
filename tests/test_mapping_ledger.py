"""Tests for Phase 3 mapping ledger contract and reconciliation."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from mimic_triggerbench.data_access import (
    MappingDecision,
    MappingLedgerRow,
    assert_mapping_ledger_consistent,
    load_all_codebooks,
    load_mapping_ledger,
    reconcile_mapping_ledger,
)


def test_load_mapping_ledger_valid() -> None:
    rows = load_mapping_ledger()
    assert len(rows) > 0
    assert all(isinstance(r, MappingLedgerRow) for r in rows)


def test_mapping_ledger_has_all_decision_classes() -> None:
    rows = load_mapping_ledger()
    decisions = {r.mapping_decision for r in rows}
    assert decisions == set(MappingDecision)


def test_mapping_ledger_retains_no_mapping_and_unsure_rows() -> None:
    rows = load_mapping_ledger()
    no_mapping = [r for r in rows if r.mapping_decision == MappingDecision.NO_MAPPING]
    unsure = [r for r in rows if r.mapping_decision == MappingDecision.UNSURE]
    assert len(no_mapping) >= 1
    assert len(unsure) >= 1
    assert all(r.canonical_concept is None for r in no_mapping + unsure)


def test_mapping_ledger_reconciliation_with_codebooks() -> None:
    rows = load_mapping_ledger()
    codebooks = load_all_codebooks()
    reconciliation = reconcile_mapping_ledger(codebooks, rows)
    assert reconciliation.ok, "\n".join(reconciliation.issue_messages)
    assert_mapping_ledger_consistent(codebooks, rows)


def test_reconciliation_catches_bad_canonical_concept() -> None:
    rows = load_mapping_ledger()
    codebooks = load_all_codebooks()
    tampered = copy.deepcopy(rows)
    first_mapped_idx = next(
        i
        for i, r in enumerate(tampered)
        if r.mapping_decision in {MappingDecision.MAP_AS_IS, MappingDecision.MAP_CONVERT_UNIT}
    )
    bad_row = tampered[first_mapped_idx].model_copy(update={"canonical_concept": "not_a_real_concept"})
    tampered[first_mapped_idx] = bad_row
    reconciliation = reconcile_mapping_ledger(codebooks, tampered)
    assert not reconciliation.ok
    assert any("not found in codebooks" in msg for msg in reconciliation.issue_messages)


def test_reject_mapped_row_without_canonical_concept() -> None:
    with pytest.raises(ValidationError):
        MappingLedgerRow(
            source_table="labevents",
            source_identifier="50971",
            source_label="Potassium",
            mapping_decision="map_as_is",
            canonical_concept=None,
            source_frequency_count=100,
            unit_frequency_summary="mEq/L:100",
            review_status="validated",
        )


def test_reject_convert_row_without_conversion_fields() -> None:
    with pytest.raises(ValidationError):
        MappingLedgerRow(
            source_table="labevents",
            source_identifier="50971",
            source_label="Potassium",
            mapping_decision="map_convert_unit",
            canonical_concept="potassium",
            source_unit="mEq/L",
            target_unit=None,
            conversion_factor=None,
            source_frequency_count=100,
            unit_frequency_summary="mEq/L:100",
            review_status="validated",
        )
