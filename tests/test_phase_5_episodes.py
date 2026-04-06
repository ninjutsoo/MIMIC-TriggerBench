"""Tests for Phase 5 — Deterministic episode and label generation.

Covers:
- Trigger evaluators for each task type with boundary conditions
- Episode model and deterministic episode_id
- Positive and negative episode generation
- No-future-leakage: negative-window future logic never stored on episodes
- Observed vs accepted action family separation
- Determinism: identical inputs → identical outputs
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta

import pytest

from mimic_triggerbench.labeling.episode_models import (
    Episode,
    NAMESPACE_EPISODE,
    generate_episode_id,
)
from mimic_triggerbench.labeling.triggers import (
    TriggerResult,
    evaluate_lab_trigger,
    evaluate_sustained_trigger,
    evaluate_trigger,
    _compare,
)
from mimic_triggerbench.labeling.task_spec_models import TriggerOperator
from mimic_triggerbench.labeling import (
    load_task_spec,
    load_all_task_specs,
    generate_episodes_for_stay,
    generate_all_episodes,
    episodes_to_records,
)
from mimic_triggerbench.timeline.models import CanonicalEvent, generate_event_uid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2150, 3, 1, 8, 0, 0)


def _ev(
    *,
    offset_min: int = 0,
    source_table: str = "labevents",
    category: str = "lab",
    name: str = "potassium",
    value: float | None = 5.0,
    unit: str | None = "mmol/L",
    stay_id: int = 100,
    subject_id: int = 10,
    hadm_id: int = 20,
    raw_id: str | None = "50971",
    event_time_end: datetime | None = None,
    metadata: dict | None = None,
) -> CanonicalEvent:
    t = T0 + timedelta(minutes=offset_min)
    uid = generate_event_uid(source_table, stay_id, t, name, raw_id=raw_id,
                             extra_discriminator=str(offset_min))
    return CanonicalEvent(
        subject_id=subject_id,
        hadm_id=hadm_id,
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
# Episode model tests
# ---------------------------------------------------------------------------

class TestEpisodeModel:
    def test_frozen(self):
        ep = Episode(
            episode_id="test",
            task_name="hyperkalemia",
            subject_id=10,
            hadm_id=20,
            stay_id=100,
            decision_time=T0,
            context_start=T0 - timedelta(hours=24),
            trigger_label=True,
            trigger_type="positive",
        )
        with pytest.raises(Exception):
            ep.subject_id = 999  # type: ignore[misc]

    def test_episode_id_deterministic(self):
        eid1 = generate_episode_id("hyperkalemia", 100, T0, "positive")
        eid2 = generate_episode_id("hyperkalemia", 100, T0, "positive")
        assert eid1 == eid2

    def test_episode_id_different_inputs(self):
        eid_a = generate_episode_id("hyperkalemia", 100, T0, "positive")
        eid_b = generate_episode_id("hypoglycemia", 100, T0, "positive")
        assert eid_a != eid_b

    def test_episode_id_is_valid_uuid(self):
        eid = generate_episode_id("hyperkalemia", 100, T0, "positive")
        parsed = uuid.UUID(eid)
        assert parsed.version == 5

    def test_namespace_is_fixed(self):
        assert NAMESPACE_EPISODE == uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


# ---------------------------------------------------------------------------
# Trigger operator tests
# ---------------------------------------------------------------------------

class TestCompare:
    def test_gte(self):
        assert _compare(6.0, TriggerOperator.GTE, 6.0)
        assert _compare(6.1, TriggerOperator.GTE, 6.0)
        assert not _compare(5.9, TriggerOperator.GTE, 6.0)

    def test_lt(self):
        assert _compare(53.0, TriggerOperator.LT, 54.0)
        assert not _compare(54.0, TriggerOperator.LT, 54.0)
        assert not _compare(55.0, TriggerOperator.LT, 54.0)


# ---------------------------------------------------------------------------
# Lab trigger tests (hyperkalemia / hypoglycemia)
# ---------------------------------------------------------------------------

class TestLabTrigger:
    def test_hyperkalemia_fires(self):
        spec = load_task_spec("hyperkalemia")
        events = [_ev(value=6.5)]
        result = evaluate_lab_trigger(spec, events, T0)
        assert result.fired is True
        assert result.signal_value == 6.5

    def test_hyperkalemia_does_not_fire_below_threshold(self):
        spec = load_task_spec("hyperkalemia")
        events = [_ev(value=5.5)]
        result = evaluate_lab_trigger(spec, events, T0)
        assert result.fired is False

    def test_hyperkalemia_boundary_exact_threshold(self):
        spec = load_task_spec("hyperkalemia")
        events = [_ev(value=6.0)]
        result = evaluate_trigger(spec, events, T0)
        assert result.fired is True

    def test_hypoglycemia_fires(self):
        spec = load_task_spec("hypoglycemia")
        events = [_ev(name="glucose", value=40.0, unit="mg/dL", raw_id="50931")]
        result = evaluate_trigger(spec, events, T0)
        assert result.fired is True

    def test_hypoglycemia_does_not_fire_at_threshold(self):
        spec = load_task_spec("hypoglycemia")
        events = [_ev(name="glucose", value=54.0, unit="mg/dL", raw_id="50931")]
        result = evaluate_trigger(spec, events, T0)
        assert result.fired is False

    def test_no_signal_means_no_trigger(self):
        spec = load_task_spec("hyperkalemia")
        result = evaluate_trigger(spec, [], T0)
        assert result.fired is False
        assert result.signal_value is None

    def test_future_event_not_used(self):
        spec = load_task_spec("hyperkalemia")
        future_event = _ev(value=7.0, offset_min=60)
        result = evaluate_trigger(spec, [future_event], T0)
        assert result.fired is False


# ---------------------------------------------------------------------------
# Sustained trigger tests (hypotension)
# ---------------------------------------------------------------------------

class TestSustainedTrigger:
    def test_hypotension_breaks_when_recovery(self):
        """MAP goes below 65 then recovers to 70: trigger should NOT fire."""
        spec = load_task_spec("hypotension")
        events = [
            _ev(name="map", value=60.0, offset_min=0, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=58.0, offset_min=10, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=70.0, offset_min=20, source_table="chartevents",
                category="vital", raw_id="220052"),
        ]
        t = T0 + timedelta(minutes=20)
        result = evaluate_sustained_trigger(spec, events, t)
        assert result.fired is False

    def test_hypotension_fires_all_low(self):
        spec = load_task_spec("hypotension")
        events = [
            _ev(name="map", value=60.0, offset_min=0, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=58.0, offset_min=10, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=55.0, offset_min=20, source_table="chartevents",
                category="vital", raw_id="220052"),
        ]
        t = T0 + timedelta(minutes=20)
        result = evaluate_sustained_trigger(spec, events, t)
        assert result.fired is True

    def test_hypotension_not_sustained_enough(self):
        spec = load_task_spec("hypotension")
        events = [
            _ev(name="map", value=60.0, offset_min=10, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=58.0, offset_min=14, source_table="chartevents",
                category="vital", raw_id="220052"),
        ]
        t = T0 + timedelta(minutes=14)
        result = evaluate_sustained_trigger(spec, events, t)
        assert result.fired is False

    def test_single_reading_not_sustained(self):
        spec = load_task_spec("hypotension")
        events = [
            _ev(name="map", value=50.0, offset_min=0, source_table="chartevents",
                category="vital", raw_id="220052"),
        ]
        result = evaluate_trigger(spec, events, T0)
        assert result.fired is False


# ---------------------------------------------------------------------------
# Episode generation tests
# ---------------------------------------------------------------------------

class TestEpisodeGeneration:
    def test_positive_episode_generated_for_hyperkalemia(self):
        spec = load_task_spec("hyperkalemia")
        events = [_ev(value=6.5)]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        positives = [e for e in episodes if e.trigger_type == "positive"]
        assert len(positives) >= 1
        assert positives[0].trigger_label is True
        assert positives[0].task_name == "hyperkalemia"
        assert len(positives[0].accepted_action_families) > 0

    def test_negative_episode_generated(self):
        spec = load_task_spec("hyperkalemia")
        events = [
            _ev(value=4.0, offset_min=0),
            _ev(value=4.2, offset_min=60),
            _ev(value=4.1, offset_min=180),
        ]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        negatives = [e for e in episodes if e.trigger_type == "negative"]
        assert len(negatives) >= 1
        assert negatives[0].trigger_label is False
        assert negatives[0].accepted_action_families == []
        assert negatives[0].observed_action_families == []

    def test_no_future_leakage_in_negative_episodes(self):
        """Negative episodes must not store any future-derived data."""
        spec = load_task_spec("hyperkalemia")
        events = [
            _ev(value=4.0, offset_min=0),
            _ev(value=4.2, offset_min=120),
        ]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        for ep in episodes:
            assert ep.decision_time <= ep.decision_time
            assert ep.context_start <= ep.decision_time
            assert not hasattr(ep, "negative_window_end")
            assert not hasattr(ep, "trigger_false_subsequent_hours")

    def test_accepted_vs_observed_separation(self):
        """Accepted (protocol) and observed (clinician) action families must be separate."""
        spec = load_task_spec("hyperkalemia")
        events = [
            _ev(value=6.5, offset_min=0),
            _ev(name="insulin_regular", source_table="inputevents", category="med_bolus",
                value=10.0, unit="units", raw_id="223257", offset_min=30),
            _ev(name="dextrose", source_table="inputevents", category="med_bolus",
                value=25.0, unit="g", raw_id="220949", offset_min=32),
        ]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        positives = [e for e in episodes if e.trigger_type == "positive"]
        assert len(positives) >= 1
        ep = positives[0]
        assert "insulin_dextrose" in ep.accepted_action_families
        assert "insulin_dextrose" in ep.observed_action_families

    def test_observed_actions_not_in_accepted_for_negatives(self):
        spec = load_task_spec("hyperkalemia")
        events = [_ev(value=4.0, offset_min=0)]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        negatives = [e for e in episodes if e.trigger_type == "negative"]
        for ep in negatives:
            assert ep.accepted_action_families == []

    def test_determinism(self):
        spec = load_task_spec("hyperkalemia")
        events = [
            _ev(value=6.5, offset_min=0),
            _ev(value=4.0, offset_min=120),
            _ev(value=4.2, offset_min=600),
        ]
        eps1 = generate_episodes_for_stay(spec, events, 100, 10, 20)
        eps2 = generate_episodes_for_stay(spec, events, 100, 10, 20)
        assert [e.episode_id for e in eps1] == [e.episode_id for e in eps2]
        assert [e.trigger_type for e in eps1] == [e.trigger_type for e in eps2]

    def test_empty_timeline_produces_no_episodes(self):
        spec = load_task_spec("hyperkalemia")
        episodes = generate_episodes_for_stay(spec, [], 100, 10, 20)
        assert episodes == []

    def test_mandatory_evidence_types_populated(self):
        spec = load_task_spec("hyperkalemia")
        events = [_ev(value=6.5)]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        for ep in episodes:
            assert len(ep.mandatory_evidence_types) > 0


# ---------------------------------------------------------------------------
# Bulk generation tests
# ---------------------------------------------------------------------------

class TestBulkGeneration:
    def test_generate_all_episodes(self):
        spec = load_task_spec("hyperkalemia")
        timelines = {
            100: [_ev(value=6.5, stay_id=100)],
            200: [_ev(value=4.0, stay_id=200, subject_id=11, hadm_id=21)],
        }
        episodes = generate_all_episodes(spec, timelines)
        stay_ids = {e.stay_id for e in episodes}
        assert 100 in stay_ids

    def test_episodes_to_records(self):
        spec = load_task_spec("hyperkalemia")
        events = [_ev(value=6.5)]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        records = episodes_to_records(episodes)
        assert len(records) == len(episodes)
        for r in records:
            assert "episode_id" in r
            assert "task_name" in r
            assert "trigger_label" in r
            assert "accepted_action_families" in r
            assert "observed_action_families" in r


# ---------------------------------------------------------------------------
# Hypotension episode generation tests
# ---------------------------------------------------------------------------

class TestHypotensionEpisodes:
    def test_positive_episode_with_sustained_low_map(self):
        spec = load_task_spec("hypotension")
        events = [
            _ev(name="map", value=60.0, offset_min=0, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=58.0, offset_min=10, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=55.0, offset_min=20, source_table="chartevents",
                category="vital", raw_id="220052"),
        ]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        positives = [e for e in episodes if e.trigger_type == "positive"]
        assert len(positives) >= 1

    def test_negative_episode_with_normal_map(self):
        spec = load_task_spec("hypotension")
        events = [
            _ev(name="map", value=75.0, offset_min=0, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=78.0, offset_min=120, source_table="chartevents",
                category="vital", raw_id="220052"),
            _ev(name="map", value=80.0, offset_min=600, source_table="chartevents",
                category="vital", raw_id="220052"),
        ]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        negatives = [e for e in episodes if e.trigger_type == "negative"]
        assert len(negatives) >= 1


# ---------------------------------------------------------------------------
# Hypoglycemia episode generation tests
# ---------------------------------------------------------------------------

class TestHypoglycemiaEpisodes:
    def test_positive_episode(self):
        spec = load_task_spec("hypoglycemia")
        events = [
            _ev(name="glucose", value=40.0, unit="mg/dL", raw_id="50931"),
        ]
        episodes = generate_episodes_for_stay(spec, events, 100, 10, 20)
        positives = [e for e in episodes if e.trigger_type == "positive"]
        assert len(positives) >= 1
        assert positives[0].trigger_label is True
