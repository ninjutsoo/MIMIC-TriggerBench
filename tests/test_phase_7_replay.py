"""Tests for Phase 7 — Replay environment and structured tools.

Covers:
- Each tool returns valid ToolResult schema
- Temporal boundary: before_time clamped to decision_time
- Future-access canary: events after decision_time never returned
- Provenance and time-range fields populated
- ReplayEnvironment dispatch, episode loading, error handling
- Tool-level filtering by category and canonical name
- Aggregation in get_trend
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from mimic_triggerbench.replay import ReplayEnvironment, ReplayEnvironmentError
from mimic_triggerbench.replay.tools import (
    get_active_infusions,
    get_recent_fluids,
    get_recent_labs,
    get_recent_meds,
    get_recent_procedures,
    get_recent_vitals,
    get_trend,
    TOOL_REGISTRY,
)
from mimic_triggerbench.schemas import (
    EpisodeInput,
    ToolResult,
    validate_tool_result,
)
from mimic_triggerbench.timeline.models import CanonicalEvent, generate_event_uid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

T0 = datetime(2150, 3, 1, 8, 0, 0)
DECISION_TIME = T0 + timedelta(hours=2)  # 10:00
STAY_ID = 100
SUBJECT_ID = 10
HADM_ID = 20


def _ev(
    *,
    offset_min: int = 0,
    source_table: str = "labevents",
    category: str = "lab",
    name: str = "potassium",
    value: float | None = 5.0,
    unit: str | None = "mmol/L",
    stay_id: int = STAY_ID,
    subject_id: int = SUBJECT_ID,
    hadm_id: int = HADM_ID,
    raw_id: str | None = "50971",
    event_time_end: datetime | None = None,
) -> CanonicalEvent:
    t = T0 + timedelta(minutes=offset_min)
    uid = generate_event_uid(
        source_table, stay_id, t, name,
        raw_id=raw_id, extra_discriminator=str(offset_min),
    )
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
    )


def _episode_input(
    *,
    stay_id: int = STAY_ID,
    decision_time: datetime = DECISION_TIME,
) -> EpisodeInput:
    return EpisodeInput(
        episode_id="test_ep_001",
        task_name="hyperkalemia",
        stay_id=stay_id,
        hadm_id=HADM_ID,
        subject_id=SUBJECT_ID,
        decision_time=decision_time,
        context_start=T0,
    )


def _sample_timeline() -> list[CanonicalEvent]:
    """Build a timeline with events before and after DECISION_TIME.

    Events at T0+0..+90 min are BEFORE decision_time (T0+120).
    Events at T0+150, T0+180 are AFTER decision_time (canaries).
    """
    return [
        _ev(offset_min=0, name="potassium", value=4.5, category="lab"),
        _ev(offset_min=30, name="potassium", value=5.8, category="lab"),
        _ev(offset_min=60, name="glucose", value=120.0, unit="mg/dL", category="lab"),
        _ev(offset_min=45, name="map", value=72.0, unit="mmHg",
            source_table="chartevents", category="vital"),
        _ev(offset_min=90, name="map", value=58.0, unit="mmHg",
            source_table="chartevents", category="vital"),
        _ev(offset_min=20, name="norepinephrine", value=0.05, unit="mcg/kg/min",
            source_table="inputevents", category="med_infusion",
            event_time_end=T0 + timedelta(hours=6)),
        _ev(offset_min=15, name="normal_saline", value=500.0, unit="mL",
            source_table="inputevents", category="med_bolus"),
        _ev(offset_min=70, name="dialysis", value=None, unit=None,
            source_table="procedureevents", category="procedure"),
        # --- CANARY future events (after DECISION_TIME = T0+120min) ---
        _ev(offset_min=150, name="potassium", value=7.2, category="lab"),
        _ev(offset_min=180, name="map", value=40.0, unit="mmHg",
            source_table="chartevents", category="vital"),
    ]


# ---------------------------------------------------------------------------
# Tool-level tests
# ---------------------------------------------------------------------------


class TestGetRecentLabs:
    def test_returns_valid_schema(self) -> None:
        events = _sample_timeline()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=["potassium"], hours_back=24,
        )
        validate_tool_result(result.model_dump(mode="json"))
        assert result.tool_name == "get_recent_labs"
        assert result.result_count == 2

    def test_filters_by_lab_name(self) -> None:
        events = _sample_timeline()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=["glucose"], hours_back=24,
        )
        assert result.result_count == 1
        assert result.results[0].canonical_name == "glucose"

    def test_no_future_events(self) -> None:
        events = _sample_timeline()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=["potassium"], hours_back=48,
        )
        for row in result.results:
            assert row.event_time <= DECISION_TIME

    def test_empty_lab_names_returns_all_labs(self) -> None:
        events = _sample_timeline()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=[], hours_back=24,
        )
        assert result.result_count == 3  # 2 potassium + 1 glucose


class TestGetRecentVitals:
    def test_returns_valid_schema(self) -> None:
        events = _sample_timeline()
        result = get_recent_vitals(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, vital_names=["map"], minutes_back=120,
        )
        validate_tool_result(result.model_dump(mode="json"))
        assert result.result_count == 2

    def test_no_future_events(self) -> None:
        events = _sample_timeline()
        result = get_recent_vitals(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, vital_names=["map"], minutes_back=300,
        )
        for row in result.results:
            assert row.event_time <= DECISION_TIME
        assert all(r.value != 40.0 for r in result.results)


class TestGetRecentMeds:
    def test_returns_valid_schema(self) -> None:
        events = _sample_timeline()
        result = get_recent_meds(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, med_classes=["norepinephrine"], hours_back=6,
        )
        validate_tool_result(result.model_dump(mode="json"))
        assert result.result_count == 1

    def test_empty_classes_returns_all_meds(self) -> None:
        events = _sample_timeline()
        result = get_recent_meds(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, med_classes=[], hours_back=6,
        )
        assert result.result_count == 2  # norepinephrine + normal_saline


class TestGetRecentFluids:
    def test_returns_valid_schema(self) -> None:
        events = _sample_timeline()
        result = get_recent_fluids(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, hours_back=6,
        )
        validate_tool_result(result.model_dump(mode="json"))
        assert result.result_count == 1
        assert result.results[0].canonical_name == "normal_saline"


class TestGetRecentProcedures:
    def test_returns_valid_schema(self) -> None:
        events = _sample_timeline()
        result = get_recent_procedures(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, hours_back=12,
        )
        validate_tool_result(result.model_dump(mode="json"))
        assert result.result_count == 1
        assert result.results[0].canonical_name == "dialysis"


class TestGetActiveInfusions:
    def test_returns_active_infusion(self) -> None:
        events = _sample_timeline()
        result = get_active_infusions(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME,
        )
        validate_tool_result(result.model_dump(mode="json"))
        assert result.result_count == 1
        assert result.results[0].canonical_name == "norepinephrine"


class TestGetTrend:
    def test_raw_aggregation(self) -> None:
        events = _sample_timeline()
        result = get_trend(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, signal_name="potassium",
            lookback_hours=6, aggregation="raw",
        )
        validate_tool_result(result.model_dump(mode="json"))
        assert result.result_count == 2

    def test_mean_aggregation(self) -> None:
        events = _sample_timeline()
        result = get_trend(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, signal_name="potassium",
            lookback_hours=6, aggregation="mean",
        )
        assert result.result_count == 1
        assert result.results[0].value == pytest.approx((4.5 + 5.8) / 2, abs=0.01)

    def test_min_max_aggregation(self) -> None:
        events = _sample_timeline()
        r_min = get_trend(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, signal_name="potassium",
            lookback_hours=6, aggregation="min",
        )
        r_max = get_trend(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, signal_name="potassium",
            lookback_hours=6, aggregation="max",
        )
        assert r_min.results[0].value == pytest.approx(4.5)
        assert r_max.results[0].value == pytest.approx(5.8)

    def test_no_future_in_trend(self) -> None:
        events = _sample_timeline()
        result = get_trend(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, signal_name="potassium",
            lookback_hours=48, aggregation="max",
        )
        assert result.results[0].value == pytest.approx(5.8)


# ---------------------------------------------------------------------------
# Temporal boundary / clamping tests
# ---------------------------------------------------------------------------


class TestTemporalClamping:
    """Verify that before_time > decision_time is clamped."""

    def test_before_time_clamped_labs(self) -> None:
        events = _sample_timeline()
        future_time = DECISION_TIME + timedelta(hours=5)
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=future_time, lab_names=["potassium"], hours_back=48,
        )
        for row in result.results:
            assert row.event_time <= DECISION_TIME
        assert result.queried_time_range.end == DECISION_TIME

    def test_before_time_clamped_vitals(self) -> None:
        events = _sample_timeline()
        future_time = DECISION_TIME + timedelta(hours=5)
        result = get_recent_vitals(
            events, decision_time=DECISION_TIME,
            before_time=future_time, vital_names=["map"], minutes_back=600,
        )
        for row in result.results:
            assert row.event_time <= DECISION_TIME

    def test_before_time_clamped_meds(self) -> None:
        events = _sample_timeline()
        future_time = DECISION_TIME + timedelta(hours=5)
        result = get_recent_meds(
            events, decision_time=DECISION_TIME,
            before_time=future_time, med_classes=[], hours_back=48,
        )
        for row in result.results:
            assert row.event_time <= DECISION_TIME

    def test_before_time_clamped_trend(self) -> None:
        events = _sample_timeline()
        future_time = DECISION_TIME + timedelta(hours=5)
        result = get_trend(
            events, decision_time=DECISION_TIME,
            before_time=future_time, signal_name="potassium",
            lookback_hours=48, aggregation="raw",
        )
        for row in result.results:
            assert row.event_time <= DECISION_TIME


# ---------------------------------------------------------------------------
# Canary test: future event NEVER appears
# ---------------------------------------------------------------------------


class TestFutureCanary:
    """Plant clearly identifiable future events and assert they never leak."""

    CANARY_VALUE = 999.99

    def _timeline_with_canary(self) -> list[CanonicalEvent]:
        """Timeline where the canary is after DECISION_TIME."""
        base = _sample_timeline()
        canary = _ev(
            offset_min=150, name="potassium", value=self.CANARY_VALUE,
            category="lab",
        )
        return base + [canary]

    def test_canary_never_in_labs(self) -> None:
        events = self._timeline_with_canary()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=["potassium"], hours_back=48,
        )
        for row in result.results:
            assert row.value != self.CANARY_VALUE

    def test_canary_never_in_trend(self) -> None:
        events = self._timeline_with_canary()
        result = get_trend(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, signal_name="potassium",
            lookback_hours=48, aggregation="max",
        )
        assert result.results[0].value != self.CANARY_VALUE

    def test_canary_never_via_environment(self) -> None:
        events = self._timeline_with_canary()
        env = ReplayEnvironment({STAY_ID: events})
        env.load_episode(_episode_input())
        result = env.call_tool("get_recent_labs", {
            "lab_names": ["potassium"], "hours_back": 48,
        })
        for row in result.results:
            assert row.value != self.CANARY_VALUE


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_source_tables_populated(self) -> None:
        events = _sample_timeline()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=["potassium"], hours_back=24,
        )
        assert "labevents" in result.provenance.source_tables

    def test_time_range_matches_query(self) -> None:
        events = _sample_timeline()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=["potassium"], hours_back=6,
        )
        assert result.queried_time_range.end == DECISION_TIME
        expected_start = DECISION_TIME - timedelta(hours=6)
        assert result.queried_time_range.start == expected_start

    def test_event_uid_populated(self) -> None:
        events = _sample_timeline()
        result = get_recent_labs(
            events, decision_time=DECISION_TIME,
            before_time=DECISION_TIME, lab_names=["potassium"], hours_back=24,
        )
        for row in result.results:
            assert row.event_uid


# ---------------------------------------------------------------------------
# ReplayEnvironment tests
# ---------------------------------------------------------------------------


class TestReplayEnvironment:
    def test_load_episode_and_call_tool(self) -> None:
        events = _sample_timeline()
        env = ReplayEnvironment({STAY_ID: events})
        env.load_episode(_episode_input())
        result = env.call_tool("get_recent_labs", {
            "lab_names": ["potassium"], "hours_back": 24,
        })
        assert isinstance(result, ToolResult)
        assert result.result_count == 2

    def test_no_episode_loaded_raises(self) -> None:
        env = ReplayEnvironment({STAY_ID: _sample_timeline()})
        with pytest.raises(ReplayEnvironmentError, match="No episode loaded"):
            env.call_tool("get_recent_labs", {"lab_names": ["potassium"], "hours_back": 24})

    def test_unknown_tool_raises(self) -> None:
        env = ReplayEnvironment({STAY_ID: _sample_timeline()})
        env.load_episode(_episode_input())
        with pytest.raises(ReplayEnvironmentError, match="Unknown tool"):
            env.call_tool("nonexistent_tool", {})

    def test_available_tools_lists_all(self) -> None:
        env = ReplayEnvironment({})
        tools = env.available_tools()
        assert "get_recent_labs" in tools
        assert "get_recent_vitals" in tools
        assert "get_trend" in tools
        assert len(tools) == len(TOOL_REGISTRY)

    def test_decision_time_injected_automatically(self) -> None:
        events = _sample_timeline()
        env = ReplayEnvironment({STAY_ID: events})
        env.load_episode(_episode_input())
        result = env.call_tool("get_recent_labs", {
            "lab_names": ["potassium"], "hours_back": 48,
        })
        for row in result.results:
            assert row.event_time <= DECISION_TIME

    def test_before_time_default_is_decision_time(self) -> None:
        events = _sample_timeline()
        env = ReplayEnvironment({STAY_ID: events})
        env.load_episode(_episode_input())
        result = env.call_tool("get_recent_labs", {
            "lab_names": ["potassium"], "hours_back": 24,
        })
        assert result.queried_time_range.end == DECISION_TIME

    def test_missing_stay_returns_empty(self) -> None:
        env = ReplayEnvironment({})
        env.load_episode(_episode_input(stay_id=9999))
        result = env.call_tool("get_recent_labs", {
            "lab_names": ["potassium"], "hours_back": 24,
        })
        assert result.result_count == 0

    def test_environment_dispatch_all_tools(self) -> None:
        """Smoke test: every registered tool can be called through the env."""
        events = _sample_timeline()
        env = ReplayEnvironment({STAY_ID: events})
        env.load_episode(_episode_input())
        tool_args = {
            "get_recent_labs": {"lab_names": ["potassium"], "hours_back": 24},
            "get_recent_vitals": {"vital_names": ["map"], "minutes_back": 120},
            "get_recent_meds": {"med_classes": [], "hours_back": 6},
            "get_recent_fluids": {"hours_back": 6},
            "get_recent_procedures": {"hours_back": 12},
            "get_active_infusions": {},
            "get_trend": {"signal_name": "potassium", "lookback_hours": 6},
        }
        for tool_name, args in tool_args.items():
            result = env.call_tool(tool_name, args)
            assert isinstance(result, ToolResult)
            validate_tool_result(result.model_dump(mode="json"))
