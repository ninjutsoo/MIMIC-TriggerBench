"""Tests for Phase 3: codebook loading, validation, normalization pipeline."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mimic_triggerbench.data_access import (
    Codebook,
    CodebookEntry,
    NormalizationResult,
    Normalizer,
    UnitConversion,
    list_codebook_domains,
    load_all_codebooks,
    load_codebook,
)
from mimic_triggerbench.labeling import load_all_task_specs

DOMAINS = ("labs", "vitals", "meds", "procedures")


# ---------------------------------------------------------------------------
# Codebook loading & structural validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_load_codebook_valid(domain: str) -> None:
    cb = load_codebook(domain)
    assert isinstance(cb, Codebook)
    assert cb.name == domain
    assert cb.version == "v0.1"
    assert len(cb.entries) >= 1


def test_load_all_codebooks() -> None:
    cbs = load_all_codebooks()
    assert set(cbs.keys()) == set(DOMAINS)
    for domain, cb in cbs.items():
        assert cb.name == domain


def test_list_codebook_domains() -> None:
    domains = list_codebook_domains()
    for d in DOMAINS:
        assert d in domains


def test_unknown_domain_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_codebook("nonexistent")


# ---------------------------------------------------------------------------
# Entry-level field checks
# ---------------------------------------------------------------------------


def test_labs_potassium_entry() -> None:
    cb = load_codebook("labs")
    entry = next(e for e in cb.entries if e.canonical_name == "potassium")
    assert 50971 in entry.raw_itemids
    assert entry.canonical_unit == "mmol/L"
    assert any(c.from_unit == "mEq/L" for c in entry.conversions)


def test_labs_glucose_entry() -> None:
    cb = load_codebook("labs")
    entry = next(e for e in cb.entries if e.canonical_name == "glucose")
    assert 50931 in entry.raw_itemids
    assert entry.canonical_unit == "mg/dL"
    assert any(c.from_unit == "mmol/L" for c in entry.conversions)


def test_vitals_map_entry() -> None:
    cb = load_codebook("vitals")
    entry = next(e for e in cb.entries if e.canonical_name == "map")
    assert 220052 in entry.raw_itemids
    assert 220181 in entry.raw_itemids
    assert entry.canonical_unit == "mmHg"


def test_meds_vasopressors_present() -> None:
    cb = load_codebook("meds")
    names = {e.canonical_name for e in cb.entries}
    for vp in ("norepinephrine", "vasopressin", "epinephrine", "phenylephrine", "dopamine"):
        assert vp in names, f"vasopressor {vp} missing from meds codebook"


def test_procedures_dialysis_present() -> None:
    cb = load_codebook("procedures")
    names = {e.canonical_name for e in cb.entries}
    assert "crrt" in names
    assert "hemodialysis" in names


# ---------------------------------------------------------------------------
# Every entry has at least one identifier
# ---------------------------------------------------------------------------


def test_all_entries_have_identifiers() -> None:
    for domain in DOMAINS:
        cb = load_codebook(domain)
        for entry in cb.entries:
            assert entry.raw_itemids or entry.raw_labels, (
                f"{domain}/{entry.canonical_name} has no identifiers"
            )


# ---------------------------------------------------------------------------
# Ambiguity tracking
# ---------------------------------------------------------------------------


def test_ambiguous_entries_flagged() -> None:
    cbs = load_all_codebooks()
    ambiguous = [
        (domain, e.canonical_name)
        for domain, cb in cbs.items()
        for e in cb.entries
        if e.is_ambiguous
    ]
    assert len(ambiguous) >= 1, "At least one entry should be flagged ambiguous"
    for _domain, _name in ambiguous:
        cb = cbs[_domain]
        entry = next(e for e in cb.entries if e.canonical_name == _name)
        assert len(entry.ambiguity_notes) >= 1, (
            f"{_domain}/{_name} is_ambiguous=True but has no ambiguity_notes"
        )


# ---------------------------------------------------------------------------
# Normalizer: itemid-based lookup
# ---------------------------------------------------------------------------


def test_normalizer_lab_by_itemid() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("labevents", itemid=50971, label=None, value=6.2, unit="mmol/L")
    assert result is not None
    assert result.canonical_name == "potassium"
    assert result.normalized_value == 6.2
    assert result.canonical_unit == "mmol/L"


def test_normalizer_vital_by_itemid() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("chartevents", itemid=220052, label=None, value=58.0, unit="mmHg")
    assert result is not None
    assert result.canonical_name == "map"
    assert result.normalized_value == 58.0


def test_normalizer_med_by_itemid() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("inputevents", itemid=221906, label=None, value=0.1)
    assert result is not None
    assert result.canonical_name == "norepinephrine"


# ---------------------------------------------------------------------------
# Normalizer: label-based lookup (fnmatch patterns)
# ---------------------------------------------------------------------------


def test_normalizer_lab_by_label() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("labevents", itemid=None, label="Potassium, Whole Blood", value=5.8, unit="mmol/L")
    assert result is not None
    assert result.canonical_name == "potassium"


def test_normalizer_med_by_label_wildcard() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("prescriptions", itemid=None, label="Glucagon 1mg IM")
    assert result is not None
    assert result.canonical_name == "glucagon"


def test_normalizer_med_by_label_kayexalate() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("prescriptions", itemid=None, label="Kayexalate 15g oral")
    assert result is not None
    assert result.canonical_name == "potassium_binder"


# ---------------------------------------------------------------------------
# Normalizer: unit conversion
# ---------------------------------------------------------------------------


def test_unit_conversion_glucose_mmol_to_mgdl() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("labevents", itemid=50931, label=None, value=3.0, unit="mmol/L")
    assert result is not None
    assert result.original_value == 3.0
    assert result.normalized_value == pytest.approx(3.0 * 18.016, rel=1e-3)
    assert result.canonical_unit == "mg/dL"


def test_unit_conversion_potassium_meq_to_mmol() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("labevents", itemid=50971, label=None, value=5.5, unit="mEq/L")
    assert result is not None
    assert result.normalized_value == pytest.approx(5.5, rel=1e-6)


def test_unit_conversion_creatinine_umol_to_mgdl() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("labevents", itemid=50912, label=None, value=100.0, unit="umol/L")
    assert result is not None
    assert result.normalized_value == pytest.approx(100.0 * 0.0113, rel=1e-3)
    assert result.canonical_unit == "mg/dL"


def test_no_conversion_when_already_canonical() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("labevents", itemid=50912, label=None, value=1.2, unit="mg/dL")
    assert result is not None
    assert result.normalized_value == 1.2


# ---------------------------------------------------------------------------
# Normalizer: unmapped rows tracked
# ---------------------------------------------------------------------------


def test_unmapped_row_returns_none_and_tracked() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("labevents", itemid=99999, label="Unknown Lab Test")
    assert result is None
    assert norm.stats.unmapped == 1
    assert 99999 in norm.stats.unmapped_itemids
    assert "Unknown Lab Test" in norm.stats.unmapped_labels


def test_stats_accumulate_correctly() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    norm.normalize("labevents", itemid=50971, label=None, value=6.0)
    norm.normalize("labevents", itemid=50971, label=None, value=6.5)
    norm.normalize("labevents", itemid=99999, label="Unknown")
    norm.normalize("labevents", itemid=99998, label="Also Unknown")
    assert norm.stats.mapped == 2
    assert norm.stats.unmapped == 2


def test_reset_stats() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    norm.normalize("labevents", itemid=50971, label=None, value=6.0)
    norm.normalize("labevents", itemid=99999, label=None)
    norm.reset_stats()
    assert norm.stats.mapped == 0
    assert norm.stats.unmapped == 0


def test_unmapped_report_generated() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    for _ in range(5):
        norm.normalize("labevents", itemid=99999, label="Obscure Lab")
    report = norm.get_unmapped_report()
    assert "Unmapped" in report
    assert "99999" in report
    assert "Obscure Lab" in report


# ---------------------------------------------------------------------------
# Normalizer: ambiguous entries correctly flagged in result
# ---------------------------------------------------------------------------


def test_ambiguous_result_flagged() -> None:
    cbs = load_all_codebooks()
    norm = Normalizer(cbs)
    result = norm.normalize("inputevents", itemid=221456, label=None)
    assert result is not None
    assert result.is_ambiguous is True
    assert len(result.ambiguity_notes) >= 1
    assert norm.stats.ambiguous >= 1


# ---------------------------------------------------------------------------
# Codebook rejection tests (malformed entries)
# ---------------------------------------------------------------------------


def test_reject_entry_with_no_identifiers() -> None:
    with pytest.raises(ValidationError):
        CodebookEntry(
            canonical_name="orphan",
            source_tables=["labevents"],
            raw_itemids=[],
            raw_labels=[],
        )


def test_reject_codebook_with_no_entries() -> None:
    with pytest.raises(ValidationError):
        Codebook(name="empty", version="v0.1", entries=[])


def test_reject_codebook_with_bad_version() -> None:
    with pytest.raises(ValidationError):
        Codebook(
            name="bad",
            version="0.1",
            entries=[
                CodebookEntry(
                    canonical_name="x",
                    source_tables=["labevents"],
                    raw_itemids=[1],
                )
            ],
        )


def test_reject_entry_with_empty_canonical_name() -> None:
    with pytest.raises(ValidationError):
        CodebookEntry(
            canonical_name="",
            source_tables=["labevents"],
            raw_itemids=[1],
        )


def test_reject_entry_with_no_source_tables() -> None:
    with pytest.raises(ValidationError):
        CodebookEntry(
            canonical_name="test",
            source_tables=[],
            raw_itemids=[1],
        )


# ---------------------------------------------------------------------------
# Coverage: task spec evidence / action families covered by codebooks
# ---------------------------------------------------------------------------

_EVIDENCE_TO_CODEBOOK = {
    "potassium_values": "potassium",
    "creatinine_bun": ("creatinine", "bun"),
    "k_lowering_meds": ("calcium_gluconate", "calcium_chloride", "insulin_regular", "sodium_bicarbonate", "potassium_binder"),
    "dialysis_procedures": ("crrt", "hemodialysis"),
    "glucose_trend": "glucose",
    "insulin_administration": "insulin_regular",
    "dextrose_glucagon": ("dextrose", "glucagon"),
    "map_trend": "map",
    "vasopressors": ("norepinephrine", "vasopressin", "epinephrine", "phenylephrine", "dopamine"),
    "iv_fluids": "crystalloid_bolus",
}


def test_task_spec_evidence_covered_by_codebooks() -> None:
    """Every required evidence type in task specs maps to at least one codebook entry."""
    cbs = load_all_codebooks()
    all_canonical = set()
    for cb in cbs.values():
        for e in cb.entries:
            all_canonical.add(e.canonical_name)

    for evidence_name, expected in _EVIDENCE_TO_CODEBOOK.items():
        if isinstance(expected, str):
            expected = (expected,)
        for canonical in expected:
            assert canonical in all_canonical, (
                f"Evidence type '{evidence_name}' expects codebook entry "
                f"'{canonical}' but it is missing."
            )


def test_task_spec_action_families_have_codebook_entries() -> None:
    """Primary action families from task specs should have matching codebook entries."""
    cbs = load_all_codebooks()
    all_canonical = set()
    for cb in cbs.values():
        for e in cb.entries:
            all_canonical.add(e.canonical_name)

    specs = load_all_task_specs()

    _ACTION_TO_CODEBOOK = {
        "calcium_administration": ("calcium_gluconate", "calcium_chloride"),
        "insulin_dextrose": ("insulin_regular", "dextrose"),
        "bicarbonate_iv": ("sodium_bicarbonate",),
        "potassium_lowering_therapy": ("potassium_binder", "albuterol_nebulized"),
        "dialysis": ("crrt", "hemodialysis"),
        "iv_dextrose": ("dextrose",),
        "glucagon": ("glucagon",),
        "fluid_bolus": ("crystalloid_bolus",),
        "vasopressor_start": ("norepinephrine", "vasopressin", "epinephrine", "phenylephrine", "dopamine"),
        "vasopressor_escalation": ("norepinephrine", "vasopressin", "epinephrine", "phenylephrine", "dopamine"),
    }

    for spec in specs.values():
        for af in spec.action_families:
            if not af.is_primary:
                continue
            expected = _ACTION_TO_CODEBOOK.get(af.name)
            if expected is None:
                continue
            for canonical in expected:
                assert canonical in all_canonical, (
                    f"Action family '{af.name}' (task={spec.task_name}) expects "
                    f"codebook entry '{canonical}' but it is missing."
                )
