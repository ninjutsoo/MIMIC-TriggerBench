"""Tests for Phase 3.5 — Action extraction feasibility checkpoint.

Covers:
- Rate normalization for vasopressors
- All four detectors: insulin+dextrose, vasopressor, fluid bolus, dialysis
- Coverage report aggregation and feasibility thresholds
- Review-set sampling
- Full checkpoint pipeline
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mimic_triggerbench.timeline.models import CanonicalEvent, generate_event_uid
from mimic_triggerbench.feasibility.detectors import (
    DetectedAction,
    detect_insulin_dextrose_pairing,
    detect_vasopressor_actions,
    detect_fluid_bolus,
    detect_dialysis_start,
    run_all_detectors,
    normalize_vaso_rate,
)
from mimic_triggerbench.feasibility.coverage_report import (
    ActionFamilyStats,
    FeasibilityThreshold,
    FeasibilityDecision,
    evaluate_feasibility,
    sample_review_set,
    write_coverage_report_md,
    write_coverage_report_json,
    run_feasibility_checkpoint,
    write_feasibility_reports,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2150, 3, 1, 8, 0, 0)


def _ev(
    *,
    offset_min: int = 0,
    source_table: str = "inputevents",
    category: str = "med_bolus",
    name: str = "insulin_regular",
    value: float | None = 10.0,
    unit: str | None = "units",
    stay_id: int = 100,
    raw_id: str | None = "223257",
    event_time_end: datetime | None = None,
    metadata: dict | None = None,
) -> CanonicalEvent:
    t = T0 + timedelta(minutes=offset_min)
    uid = generate_event_uid(source_table, stay_id, t, name, raw_id=raw_id,
                             extra_discriminator=str(offset_min))
    return CanonicalEvent(
        subject_id=10,
        hadm_id=20,
        stay_id=stay_id,
        event_time=t,
        event_time_end=event_time_end,
        event_uid=uid,
        source_table=source_table,
        event_category=category,
        canonical_name=name,
        value_num=value,
        unit=unit,
        raw_id=raw_id,
        metadata_json=json.dumps(metadata or {}, default=str),
    )


# ---------------------------------------------------------------------------
# Rate normalizer tests
# ---------------------------------------------------------------------------

class TestRateNormalizer:
    def test_norepinephrine_mcg_kg_min_passthrough(self):
        rate, unit = normalize_vaso_rate("norepinephrine", 0.1, "mcg/kg/min")
        assert rate == 0.1
        assert unit == "mcg/kg/min"

    def test_norepinephrine_mg_kg_min_conversion(self):
        rate, unit = normalize_vaso_rate("norepinephrine", 0.001, "mg/kg/min")
        assert rate == pytest.approx(1.0)
        assert unit == "mcg/kg/min"

    def test_norepinephrine_mcg_min_with_weight(self):
        rate, unit = normalize_vaso_rate("norepinephrine", 8.0, "mcg/min", patient_weight_kg=80.0)
        assert rate == pytest.approx(0.1)
        assert unit == "mcg/kg/min"

    def test_norepinephrine_mcg_min_without_weight(self):
        rate, unit = normalize_vaso_rate("norepinephrine", 8.0, "mcg/min")
        assert rate == 8.0
        assert unit == "mcg/min"

    def test_vasopressin_units_min_to_hr(self):
        rate, unit = normalize_vaso_rate("vasopressin", 0.04, "units/min")
        assert rate == pytest.approx(2.4)
        assert unit == "units/hr"

    def test_vasopressin_units_hr_passthrough(self):
        rate, unit = normalize_vaso_rate("vasopressin", 2.4, "units/hr")
        assert rate == 2.4
        assert unit == "units/hr"

    def test_phenylephrine_mcg_min_passthrough(self):
        rate, unit = normalize_vaso_rate("phenylephrine", 100.0, "mcg/min")
        assert rate == 100.0
        assert unit == "mcg/min"

    def test_phenylephrine_mg_min_conversion(self):
        rate, unit = normalize_vaso_rate("phenylephrine", 0.1, "mg/min")
        assert rate == pytest.approx(100.0)
        assert unit == "mcg/min"

    def test_none_rate_returns_none(self):
        rate, unit = normalize_vaso_rate("norepinephrine", None, None)
        assert rate is None

    def test_unknown_drug(self):
        rate, unit = normalize_vaso_rate("unknown_drug", 1.0, "mg/min")
        assert unit == "unknown"

    def test_mg_kg_min_sentinel_weight(self):
        """Direct-quote: patientweight = 1 sentinel → pass rate as-is."""
        rate, unit = normalize_vaso_rate("norepinephrine", 0.05, "mg/kg/min", patient_weight_kg=1.0)
        assert rate == 0.05
        assert unit == "mcg/kg/min"


# ---------------------------------------------------------------------------
# Insulin + dextrose pairing detector tests
# ---------------------------------------------------------------------------

class TestInsulinDextrosePairing:
    def test_paired_within_window(self):
        events = [
            _ev(name="insulin_regular", offset_min=0),
            _ev(name="dextrose", offset_min=30, raw_id="220949"),
        ]
        detected = detect_insulin_dextrose_pairing(events)
        assert len(detected) == 1
        assert detected[0].action_family == "insulin_dextrose"
        assert detected[0].details["gap_minutes"] == 30.0

    def test_not_paired_outside_window(self):
        events = [
            _ev(name="insulin_regular", offset_min=0),
            _ev(name="dextrose", offset_min=180, raw_id="220949"),
        ]
        detected = detect_insulin_dextrose_pairing(events, pairing_window_hours=1.0)
        assert len(detected) == 0

    def test_no_dextrose_means_no_pair(self):
        events = [_ev(name="insulin_regular", offset_min=0)]
        detected = detect_insulin_dextrose_pairing(events)
        assert len(detected) == 0

    def test_dextrose_used_only_once(self):
        events = [
            _ev(name="insulin_regular", offset_min=0),
            _ev(name="insulin_regular", offset_min=10, raw_id="223258"),
            _ev(name="dextrose", offset_min=5, raw_id="220949"),
        ]
        detected = detect_insulin_dextrose_pairing(events)
        assert len(detected) == 1

    def test_provenance_uids_tracked(self):
        events = [
            _ev(name="insulin_regular", offset_min=0),
            _ev(name="dextrose", offset_min=15, raw_id="220949"),
        ]
        detected = detect_insulin_dextrose_pairing(events)
        assert len(detected[0].source_event_uids) == 2


# ---------------------------------------------------------------------------
# Vasopressor start / escalation detector tests
# ---------------------------------------------------------------------------

class TestVasopressorDetector:
    def _infusion(self, *, name: str = "norepinephrine", offset_min: int = 0,
                  rate: float = 0.1, rate_unit: str = "mcg/kg/min",
                  end_offset_min: int | None = None, stay_id: int = 100) -> CanonicalEvent:
        t_end = T0 + timedelta(minutes=end_offset_min) if end_offset_min else None
        return _ev(
            name=name,
            offset_min=offset_min,
            category="med_infusion",
            source_table="inputevents",
            raw_id="221906",
            value=2.0,
            unit="mg",
            event_time_end=t_end,
            stay_id=stay_id,
            metadata={"rate": rate, "rate_unit": rate_unit},
        )

    def test_first_infusion_is_start(self):
        events = [self._infusion(offset_min=0)]
        detected = detect_vasopressor_actions(events)
        starts = [d for d in detected if d.action_family == "vasopressor_start"]
        assert len(starts) == 1

    def test_escalation_detected(self):
        events = [
            self._infusion(offset_min=0, rate=0.05),
            self._infusion(offset_min=60, rate=0.10),
        ]
        detected = detect_vasopressor_actions(events)
        escalations = [d for d in detected if d.action_family == "vasopressor_escalation"]
        assert len(escalations) == 1
        assert escalations[0].details["pct_increase"] == pytest.approx(100.0)

    def test_no_escalation_below_threshold(self):
        events = [
            self._infusion(offset_min=0, rate=0.10),
            self._infusion(offset_min=60, rate=0.11),
        ]
        detected = detect_vasopressor_actions(events)
        escalations = [d for d in detected if d.action_family == "vasopressor_escalation"]
        assert len(escalations) == 0

    def test_restart_after_gap(self):
        events = [
            self._infusion(offset_min=0, rate=0.1, end_offset_min=60),
            self._infusion(offset_min=120, rate=0.05),
        ]
        detected = detect_vasopressor_actions(events)
        starts = [d for d in detected if d.action_family == "vasopressor_start"]
        assert len(starts) == 2

    def test_rate_normalization_applied(self):
        events = [self._infusion(offset_min=0, rate=0.001, rate_unit="mg/kg/min")]
        detected = detect_vasopressor_actions(events)
        assert detected[0].details["rate"] == pytest.approx(1.0)

    def test_multiple_drugs_tracked_independently(self):
        events = [
            self._infusion(name="norepinephrine", offset_min=0),
            self._infusion(name="vasopressin", offset_min=10, rate=0.04, rate_unit="units/min"),
        ]
        detected = detect_vasopressor_actions(events)
        starts = [d for d in detected if d.action_family == "vasopressor_start"]
        drugs = {s.details["drug"] for s in starts}
        assert drugs == {"norepinephrine", "vasopressin"}


# ---------------------------------------------------------------------------
# Fluid bolus detector tests
# ---------------------------------------------------------------------------

class TestFluidBolusDetector:
    def test_bolus_by_category(self):
        ev = _ev(
            name="crystalloid_bolus", category="med_bolus",
            value=500.0, unit="mL", raw_id="225158",
        )
        detected = detect_fluid_bolus([ev])
        assert len(detected) == 1
        assert detected[0].details["is_bolus_by_category"] is True

    def test_bolus_by_volume(self):
        ev = _ev(
            name="crystalloid_bolus", category="med_infusion",
            value=500.0, unit="mL", raw_id="225158",
        )
        detected = detect_fluid_bolus([ev])
        assert len(detected) == 1

    def test_small_volume_excluded(self):
        ev = _ev(
            name="crystalloid_bolus", category="med_infusion",
            value=100.0, unit="mL", raw_id="225158",
        )
        detected = detect_fluid_bolus([ev])
        assert len(detected) == 0

    def test_rate_based_bolus(self):
        t_end = T0 + timedelta(minutes=20)
        ev = _ev(
            name="crystalloid_bolus", category="med_infusion",
            value=500.0, unit="mL", raw_id="225158",
            event_time_end=t_end,
        )
        detected = detect_fluid_bolus([ev])
        assert len(detected) == 1
        assert detected[0].details["is_bolus_by_rate"] is True


# ---------------------------------------------------------------------------
# Dialysis start detector tests
# ---------------------------------------------------------------------------

class TestDialysisStartDetector:
    def test_single_crrt_start(self):
        ev = _ev(
            name="crrt", source_table="procedureevents",
            category="procedure", raw_id="225441",
            event_time_end=T0 + timedelta(hours=4),
        )
        detected = detect_dialysis_start([ev])
        assert len(detected) == 1
        assert detected[0].details["modality"] == "crrt"

    def test_dedup_within_4h(self):
        events = [
            _ev(name="crrt", source_table="procedureevents",
                category="procedure", raw_id="225441", offset_min=0),
            _ev(name="crrt", source_table="procedureevents",
                category="procedure", raw_id="225441", offset_min=60),
        ]
        detected = detect_dialysis_start(events)
        assert len(detected) == 1

    def test_separate_modalities(self):
        events = [
            _ev(name="crrt", source_table="procedureevents",
                category="procedure", raw_id="225441", offset_min=0),
            _ev(name="hemodialysis", source_table="procedureevents",
                category="procedure", raw_id="225436", offset_min=30),
        ]
        detected = detect_dialysis_start(events)
        assert len(detected) == 2

    def test_restart_after_gap(self):
        events = [
            _ev(name="crrt", source_table="procedureevents",
                category="procedure", raw_id="225441", offset_min=0),
            _ev(name="crrt", source_table="procedureevents",
                category="procedure", raw_id="225441", offset_min=300),
        ]
        detected = detect_dialysis_start(events)
        assert len(detected) == 2


# ---------------------------------------------------------------------------
# run_all_detectors integration test
# ---------------------------------------------------------------------------

class TestRunAllDetectors:
    def test_all_families_detected(self):
        events = [
            _ev(name="insulin_regular", offset_min=0),
            _ev(name="dextrose", offset_min=15, raw_id="220949"),
            _ev(name="norepinephrine", category="med_infusion",
                offset_min=30, raw_id="221906",
                metadata={"rate": 0.1, "rate_unit": "mcg/kg/min"}),
            _ev(name="crystalloid_bolus", category="med_bolus",
                value=500.0, unit="mL", offset_min=45, raw_id="225158"),
            _ev(name="crrt", source_table="procedureevents",
                category="procedure", offset_min=60, raw_id="225441"),
        ]
        results = run_all_detectors(events)
        assert "insulin_dextrose" in results
        assert "vasopressor" in results
        assert "fluid_bolus" in results
        assert "dialysis_start" in results

    def test_empty_timeline(self):
        results = run_all_detectors([])
        for family, actions in results.items():
            assert actions == []


# ---------------------------------------------------------------------------
# Coverage report and feasibility tests
# ---------------------------------------------------------------------------

class TestActionFamilyStats:
    def test_add_detections(self):
        stats = ActionFamilyStats(action_family="test")
        actions = [
            DetectedAction("test", 100, 10, 20, T0, None, {}, ()),
            DetectedAction("test", 100, 10, 20, T0, None, {}, ()),
            DetectedAction("test", 200, 11, 21, T0, None, {}, ()),
        ]
        stats.add_detections(actions)
        assert stats.detected_events == 3
        assert stats.unique_stays == 2
        assert stats.unique_patients == 2

    def test_to_dict(self):
        stats = ActionFamilyStats(action_family="test")
        d = stats.to_dict()
        assert d["action_family"] == "test"
        assert d["detected_events"] == 0


class TestFeasibilityThreshold:
    def test_pass(self):
        t = FeasibilityThreshold("test", min_detected_events=2, min_unique_stays=1, min_unique_patients=1)
        stats = ActionFamilyStats(action_family="test")
        stats.add_detections([
            DetectedAction("test", 100, 10, 20, T0, None, {}, ()),
            DetectedAction("test", 200, 11, 21, T0, None, {}, ()),
            DetectedAction("test", 300, 12, 22, T0, None, {}, ()),
        ])
        assert t.evaluate(stats) is True

    def test_fail(self):
        t = FeasibilityThreshold("test", min_detected_events=100)
        stats = ActionFamilyStats(action_family="test")
        stats.add_detections([DetectedAction("test", 100, 10, 20, T0, None, {}, ())])
        assert t.evaluate(stats) is False


class TestEvaluateFeasibility:
    def test_produces_decisions(self):
        stats = {"test_family": ActionFamilyStats(action_family="test_family")}
        thresholds = {"test_family": FeasibilityThreshold("test_family", min_detected_events=0, min_unique_stays=0, min_unique_patients=0)}
        decisions = evaluate_feasibility(stats, thresholds)
        assert len(decisions) == 1
        assert decisions[0].passed is True


class TestSampleReviewSet:
    def test_deterministic(self):
        actions = [
            DetectedAction(f"test", i, 10, 20, T0, None, {}, ())
            for i in range(50)
        ]
        s1 = sample_review_set(actions, n=10, seed=42)
        s2 = sample_review_set(actions, n=10, seed=42)
        assert [a.stay_id for a in s1] == [a.stay_id for a in s2]

    def test_full_if_under_n(self):
        actions = [DetectedAction("test", 1, 10, 20, T0, None, {}, ())]
        result = sample_review_set(actions, n=25)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Report writing tests
# ---------------------------------------------------------------------------

class TestReportWriting:
    def test_write_md_report(self, tmp_path: Path):
        decisions = [FeasibilityDecision(
            action_family="test",
            threshold=FeasibilityThreshold("test"),
            stats=ActionFamilyStats(action_family="test"),
            passed=True,
            note="OK",
        )]
        write_coverage_report_md(decisions, {}, tmp_path / "report.md")
        content = (tmp_path / "report.md").read_text()
        assert "test" in content
        assert "PASS" in content

    def test_write_json_report(self, tmp_path: Path):
        decisions = [FeasibilityDecision(
            action_family="test",
            threshold=FeasibilityThreshold("test"),
            stats=ActionFamilyStats(action_family="test"),
            passed=False,
            note="below",
        )]
        write_coverage_report_json(decisions, {}, tmp_path / "report.json")
        data = json.loads((tmp_path / "report.json").read_text())
        assert data["decisions"][0]["passed"] is False

    def test_write_feasibility_reports(self, tmp_path: Path):
        decisions = [FeasibilityDecision(
            action_family="test",
            threshold=FeasibilityThreshold("test"),
            stats=ActionFamilyStats(action_family="test"),
            passed=True,
        )]
        write_feasibility_reports(decisions, {}, tmp_path)
        assert (tmp_path / "feasibility_report.md").exists()
        assert (tmp_path / "feasibility_report.json").exists()


# ---------------------------------------------------------------------------
# Full checkpoint pipeline test
# ---------------------------------------------------------------------------

class TestCheckpointPipeline:
    def test_full_pipeline_with_synthetic_timelines(self):
        events = [
            _ev(name="insulin_regular", offset_min=0),
            _ev(name="dextrose", offset_min=15, raw_id="220949"),
            _ev(name="norepinephrine", category="med_infusion",
                offset_min=30, raw_id="221906",
                metadata={"rate": 0.1, "rate_unit": "mcg/kg/min"}),
            _ev(name="crystalloid_bolus", category="med_bolus",
                value=500.0, unit="mL", offset_min=45, raw_id="225158"),
            _ev(name="crrt", source_table="procedureevents",
                category="procedure", offset_min=60, raw_id="225441"),
        ]
        timelines = {100: events}

        low_thresholds = {
            family: FeasibilityThreshold(family, min_detected_events=1, min_unique_stays=1, min_unique_patients=1)
            for family in ("insulin_dextrose", "vasopressor_start", "vasopressor_escalation",
                          "fluid_bolus", "dialysis_start")
        }

        decisions, review_sets = run_feasibility_checkpoint(
            timelines, thresholds=low_thresholds,
        )
        assert len(decisions) >= 4
        passed = {d.action_family for d in decisions if d.passed}
        assert "insulin_dextrose" in passed
        assert "vasopressor_start" in passed
        assert "fluid_bolus" in passed
        assert "dialysis_start" in passed

    def test_pipeline_writes_reports(self, tmp_path: Path):
        events = [
            _ev(name="insulin_regular", offset_min=0),
            _ev(name="dextrose", offset_min=15, raw_id="220949"),
        ]
        timelines = {100: events}
        decisions, review_sets = run_feasibility_checkpoint(timelines)
        write_feasibility_reports(decisions, review_sets, tmp_path)
        assert (tmp_path / "feasibility_report.md").exists()
        assert (tmp_path / "feasibility_report.json").exists()

    def test_empty_timelines(self):
        decisions, review_sets = run_feasibility_checkpoint({})
        assert all(not d.passed for d in decisions)


# ---------------------------------------------------------------------------
# DetectedAction serialization test
# ---------------------------------------------------------------------------

class TestDetectedAction:
    def test_to_dict(self):
        a = DetectedAction(
            action_family="test",
            stay_id=100,
            subject_id=10,
            hadm_id=20,
            detection_time=T0,
            detection_time_end=T0 + timedelta(hours=1),
            details={"key": "val"},
            source_event_uids=("uid1", "uid2"),
        )
        d = a.to_dict()
        assert d["action_family"] == "test"
        assert d["detection_time"] == T0.isoformat()
        assert len(d["source_event_uids"]) == 2
