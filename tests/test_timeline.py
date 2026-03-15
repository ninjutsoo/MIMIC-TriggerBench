"""Tests for Phase 4 — Canonical timeline construction.

Covers:
- CanonicalEvent model and deterministic sort key
- event_uid generation and reproducibility
- Timeline builder with synthetic source DataFrames
- As-of time slicing and filtering utilities
- Parquet round-trip I/O
- Ordering determinism across rebuilds
- Provenance preservation in metadata_json
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from mimic_triggerbench.timeline.models import CanonicalEvent, events_sorted, generate_event_uid, NAMESPACE_TIMELINE
from mimic_triggerbench.timeline.builder import (
    build_stay_timeline,
    build_stay_lookup,
    _classify_input_event,
)
from mimic_triggerbench.timeline.slicing import (
    slice_as_of,
    slice_window,
    slice_lookback,
    filter_by_category,
    filter_by_canonical_name,
    filter_by_source_table,
    get_latest_value,
    get_recent_values,
    active_infusions_at,
)
from mimic_triggerbench.timeline.io import (
    events_to_dataframe,
    dataframe_to_events,
    write_timeline_parquet,
    read_timeline_parquet,
)
from mimic_triggerbench.data_access.codebook_loader import load_all_codebooks
from mimic_triggerbench.data_access.normalizer import Normalizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2150, 3, 1, 8, 0, 0)
T1 = datetime(2150, 3, 1, 9, 0, 0)
T2 = datetime(2150, 3, 1, 10, 0, 0)
T3 = datetime(2150, 3, 1, 11, 0, 0)


def _make_event(
    *,
    event_time: datetime = T0,
    event_time_end: datetime | None = None,
    source_table: str = "chartevents",
    event_category: str = "vital",
    canonical_name: str = "map",
    value_num: float | None = 65.0,
    unit: str | None = "mmHg",
    stay_id: int = 30000001,
    event_uid: str | None = None,
    raw_id: str | None = "220052",
    raw_label: str | None = "Arterial Blood Pressure mean",
    metadata_json: str = "{}",
) -> CanonicalEvent:
    if event_uid is None:
        event_uid = generate_event_uid(
            source_table, stay_id, event_time, canonical_name, raw_id=raw_id,
        )
    return CanonicalEvent(
        subject_id=10000001,
        hadm_id=20000001,
        stay_id=stay_id,
        event_time=event_time,
        event_time_end=event_time_end,
        event_uid=event_uid,
        source_table=source_table,
        event_category=event_category,
        canonical_name=canonical_name,
        value_num=value_num,
        unit=unit,
        raw_id=raw_id,
        raw_label=raw_label,
        metadata_json=metadata_json,
    )


@pytest.fixture()
def normalizer() -> Normalizer:
    return Normalizer(load_all_codebooks())


@pytest.fixture()
def sample_timeline() -> list[CanonicalEvent]:
    return [
        _make_event(event_time=T0, canonical_name="map", value_num=60.0),
        _make_event(event_time=T0, source_table="labevents", event_category="lab",
                     canonical_name="potassium", value_num=6.5, unit="mmol/L",
                     raw_id="50971"),
        _make_event(event_time=T1, canonical_name="map", value_num=55.0),
        _make_event(event_time=T2, source_table="inputevents", event_category="med_infusion",
                     canonical_name="norepinephrine", value_num=0.1,
                     unit="mcg/kg/min", raw_id="221906",
                     event_time_end=T3),
        _make_event(event_time=T3, canonical_name="map", value_num=70.0),
    ]


# ---------------------------------------------------------------------------
# CanonicalEvent model tests
# ---------------------------------------------------------------------------

class TestCanonicalEvent:
    def test_frozen(self):
        ev = _make_event()
        with pytest.raises(Exception):
            ev.subject_id = 999  # type: ignore[misc]

    def test_sort_key_components(self):
        ev = _make_event()
        sk = ev.sort_key
        assert len(sk) == 5
        assert sk[0] == ev.event_time

    def test_metadata_dict_round_trip(self):
        meta = {"original_value": 5.5, "codebook_version": "v0.1"}
        ev = _make_event(metadata_json=json.dumps(meta))
        assert ev.metadata_dict() == meta


# ---------------------------------------------------------------------------
# event_uid tests
# ---------------------------------------------------------------------------

class TestEventUid:
    def test_deterministic(self):
        uid1 = generate_event_uid("labevents", 100, T0, "potassium", raw_id="50971")
        uid2 = generate_event_uid("labevents", 100, T0, "potassium", raw_id="50971")
        assert uid1 == uid2

    def test_different_inputs_different_uids(self):
        uid_a = generate_event_uid("labevents", 100, T0, "potassium", raw_id="50971")
        uid_b = generate_event_uid("labevents", 100, T0, "glucose", raw_id="50931")
        assert uid_a != uid_b

    def test_is_valid_uuid(self):
        uid = generate_event_uid("labevents", 100, T0, "potassium")
        parsed = uuid.UUID(uid)
        assert parsed.version == 5

    def test_extra_discriminator_creates_distinct_uid(self):
        uid1 = generate_event_uid("labevents", 100, T0, "potassium", extra_discriminator="")
        uid2 = generate_event_uid("labevents", 100, T0, "potassium", extra_discriminator="1")
        assert uid1 != uid2

    def test_namespace_is_fixed(self):
        assert NAMESPACE_TIMELINE == uuid.UUID("7a3c9f1e-4d2b-4e8a-b5c6-1f2e3d4a5b6c")


# ---------------------------------------------------------------------------
# Ordering determinism tests
# ---------------------------------------------------------------------------

class TestOrderingDeterminism:
    def test_events_sorted_is_stable(self, sample_timeline: list[CanonicalEvent]):
        sorted1 = events_sorted(sample_timeline)
        sorted2 = events_sorted(list(reversed(sample_timeline)))
        uids1 = [e.event_uid for e in sorted1]
        uids2 = [e.event_uid for e in sorted2]
        assert uids1 == uids2

    def test_same_timestamp_tie_break_by_category(self):
        lab = _make_event(event_time=T0, event_category="lab", canonical_name="potassium",
                          source_table="labevents", raw_id="50971")
        vital = _make_event(event_time=T0, event_category="vital", canonical_name="map",
                            source_table="chartevents", raw_id="220052")
        sorted_events = events_sorted([vital, lab])
        assert sorted_events[0].event_category == "lab"
        assert sorted_events[1].event_category == "vital"

    def test_same_timestamp_same_category_tie_break_by_name(self):
        ev_a = _make_event(event_time=T0, canonical_name="heart_rate", raw_id="220045")
        ev_b = _make_event(event_time=T0, canonical_name="map", raw_id="220052")
        sorted_events = events_sorted([ev_b, ev_a])
        assert sorted_events[0].canonical_name == "heart_rate"
        assert sorted_events[1].canonical_name == "map"

    def test_rebuild_produces_identical_uids(self, normalizer: Normalizer):
        """Simulates two independent builds from the same source data."""
        chart_df = pd.DataFrame({
            "subject_id": [10],
            "hadm_id": [20],
            "stay_id": [30],
            "charttime": [pd.Timestamp("2150-03-01 08:00:00")],
            "itemid": [220052],
            "valuenum": [65.0],
            "valueuom": ["mmHg"],
        })
        tl1 = build_stay_timeline(30, chartevents=chart_df, normalizer=normalizer)
        normalizer.reset_stats()
        tl2 = build_stay_timeline(30, chartevents=chart_df, normalizer=normalizer)
        assert len(tl1) == len(tl2) == 1
        assert tl1[0].event_uid == tl2[0].event_uid


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------

class TestBuilder:
    def test_build_chartevents(self, normalizer: Normalizer):
        df = pd.DataFrame({
            "subject_id": [10, 10],
            "hadm_id": [20, 20],
            "stay_id": [30, 30],
            "charttime": pd.to_datetime(["2150-03-01 08:00", "2150-03-01 09:00"]),
            "itemid": [220052, 220045],
            "valuenum": [65.0, 80.0],
            "valueuom": ["mmHg", "bpm"],
        })
        tl = build_stay_timeline(30, chartevents=df, normalizer=normalizer)
        assert len(tl) == 2
        names = {e.canonical_name for e in tl}
        assert "map" in names
        assert "heart_rate" in names

    def test_build_labevents_with_stay_lookup(self, normalizer: Normalizer):
        lab_df = pd.DataFrame({
            "subject_id": [10],
            "hadm_id": [20],
            "charttime": pd.to_datetime(["2150-03-01 08:30"]),
            "itemid": [50971],
            "valuenum": [6.2],
            "valueuom": ["mmol/L"],
        })
        lookup = {(10, 20): 30}
        tl = build_stay_timeline(
            30, labevents=lab_df, normalizer=normalizer, stay_lookup=lookup,
        )
        assert len(tl) == 1
        assert tl[0].canonical_name == "potassium"
        assert tl[0].stay_id == 30

    def test_build_inputevents_interval(self, normalizer: Normalizer):
        df = pd.DataFrame({
            "subject_id": [10],
            "hadm_id": [20],
            "stay_id": [30],
            "starttime": pd.to_datetime(["2150-03-01 10:00"]),
            "endtime": pd.to_datetime(["2150-03-01 12:00"]),
            "itemid": [221906],
            "amount": [2.0],
            "amountuom": ["mg"],
            "rate": [0.1],
            "rateuom": ["mcg/kg/min"],
            "ordercategorydescription": ["Continuous Med"],
            "ordercategoryname": ["Continuous Med"],
            "orderid": [1001],
            "linkorderid": [1001],
        })
        tl = build_stay_timeline(30, inputevents=df, normalizer=normalizer)
        assert len(tl) == 1
        ev = tl[0]
        assert ev.canonical_name == "norepinephrine"
        assert ev.event_time_end is not None
        assert ev.event_category == "med_infusion"
        meta = ev.metadata_dict()
        assert meta["order_id"] == 1001
        assert meta["link_order_id"] == 1001

    def test_build_procedureevents(self, normalizer: Normalizer):
        df = pd.DataFrame({
            "subject_id": [10],
            "hadm_id": [20],
            "stay_id": [30],
            "starttime": pd.to_datetime(["2150-03-01 09:00"]),
            "endtime": pd.to_datetime(["2150-03-01 13:00"]),
            "itemid": [225441],
        })
        tl = build_stay_timeline(30, procedureevents=df, normalizer=normalizer)
        assert len(tl) == 1
        assert tl[0].event_category == "procedure"
        assert tl[0].canonical_name == "crrt"
        assert tl[0].event_time_end is not None

    def test_build_stay_lookup(self):
        icu_df = pd.DataFrame({
            "subject_id": [10, 10, 20],
            "hadm_id": [20, 21, 30],
            "stay_id": [100, 200, 300],
            "intime": pd.to_datetime(["2150-01-01", "2150-02-01", "2150-01-15"]),
            "outtime": pd.to_datetime(["2150-01-05", "2150-02-05", "2150-01-20"]),
        })
        lookup = build_stay_lookup(icu_df)
        assert lookup[(10, 20)] == 100
        assert lookup[(10, 21)] == 200
        assert lookup[(20, 30)] == 300

    def test_classify_bolus(self):
        class _Row:
            ordercategorydescription = "02-Bolus"
            ordercategoryname = "Bolus"
            rate = 0.0
        assert _classify_input_event(_Row()) == "med_bolus"

    def test_classify_continuous(self):
        class _Row:
            ordercategorydescription = "Continuous Med"
            ordercategoryname = "Continuous Med"
            rate = 0.05
        assert _classify_input_event(_Row()) == "med_infusion"

    def test_empty_sources_produce_empty_timeline(self, normalizer: Normalizer):
        tl = build_stay_timeline(30, normalizer=normalizer)
        assert tl == []

    def test_provenance_preserved_in_metadata(self, normalizer: Normalizer):
        df = pd.DataFrame({
            "subject_id": [10],
            "hadm_id": [20],
            "stay_id": [30],
            "charttime": pd.to_datetime(["2150-03-01 08:00"]),
            "itemid": [220052],
            "valuenum": [65.0],
            "valueuom": ["mmHg"],
        })
        tl = build_stay_timeline(30, chartevents=df, normalizer=normalizer)
        meta = tl[0].metadata_dict()
        assert "codebook_version" in meta
        assert meta["codebook_version"] == "v0.1"


# ---------------------------------------------------------------------------
# As-of slicing tests
# ---------------------------------------------------------------------------

class TestSlicing:
    def test_slice_as_of_excludes_future(self, sample_timeline: list[CanonicalEvent]):
        result = slice_as_of(sample_timeline, T1)
        for e in result:
            assert e.event_time <= T1
        assert any(e.event_time == T1 for e in result)

    def test_slice_as_of_includes_boundary(self, sample_timeline: list[CanonicalEvent]):
        result = slice_as_of(sample_timeline, T0)
        assert all(e.event_time == T0 for e in result)

    def test_slice_window(self, sample_timeline: list[CanonicalEvent]):
        result = slice_window(sample_timeline, T1, T2)
        for e in result:
            assert T1 <= e.event_time <= T2

    def test_slice_lookback(self, sample_timeline: list[CanonicalEvent]):
        result = slice_lookback(sample_timeline, T2, hours_back=1.0)
        for e in result:
            assert e.event_time >= T2 - timedelta(hours=1)
            assert e.event_time <= T2

    def test_filter_by_category(self, sample_timeline: list[CanonicalEvent]):
        result = filter_by_category(sample_timeline, {"lab"})
        assert all(e.event_category == "lab" for e in result)
        assert len(result) >= 1

    def test_filter_by_canonical_name(self, sample_timeline: list[CanonicalEvent]):
        result = filter_by_canonical_name(sample_timeline, {"map"})
        assert all(e.canonical_name == "map" for e in result)

    def test_filter_by_source_table(self, sample_timeline: list[CanonicalEvent]):
        result = filter_by_source_table(sample_timeline, {"labevents"})
        assert all(e.source_table == "labevents" for e in result)

    def test_get_latest_value(self, sample_timeline: list[CanonicalEvent]):
        latest = get_latest_value(sample_timeline, "map", T2)
        assert latest is not None
        assert latest.event_time == T1

    def test_get_latest_value_none_if_no_match(self, sample_timeline: list[CanonicalEvent]):
        assert get_latest_value(sample_timeline, "nonexistent", T3) is None

    def test_get_recent_values(self, sample_timeline: list[CanonicalEvent]):
        result = get_recent_values(sample_timeline, "map", T3, hours_back=3.0)
        assert all(e.canonical_name == "map" for e in result)

    def test_active_infusions_at(self, sample_timeline: list[CanonicalEvent]):
        active = active_infusions_at(sample_timeline, T2 + timedelta(minutes=30))
        assert len(active) == 1
        assert active[0].canonical_name == "norepinephrine"

    def test_active_infusions_after_end(self, sample_timeline: list[CanonicalEvent]):
        active = active_infusions_at(sample_timeline, T3 + timedelta(hours=1))
        assert len(active) == 0

    def test_future_access_blocked(self, sample_timeline: list[CanonicalEvent]):
        """As-of slicing must never return events after the query time."""
        result = slice_as_of(sample_timeline, T0)
        future_events = [e for e in result if e.event_time > T0]
        assert future_events == [], "Future events leaked through as-of slice!"


# ---------------------------------------------------------------------------
# Parquet I/O tests
# ---------------------------------------------------------------------------

class TestParquetIO:
    def test_round_trip(self, sample_timeline: list[CanonicalEvent], tmp_path: Path):
        timelines = {30000001: sample_timeline}
        write_timeline_parquet(timelines, tmp_path)
        loaded = read_timeline_parquet(tmp_path)
        assert 30000001 in loaded
        original_uids = [e.event_uid for e in sample_timeline]
        loaded_uids = [e.event_uid for e in loaded[30000001]]
        assert set(original_uids) == set(loaded_uids)

    def test_round_trip_preserves_values(self, sample_timeline: list[CanonicalEvent], tmp_path: Path):
        timelines = {30000001: sample_timeline}
        write_timeline_parquet(timelines, tmp_path)
        loaded = read_timeline_parquet(tmp_path)
        for orig, loaded_ev in zip(
            events_sorted(sample_timeline),
            events_sorted(loaded[30000001]),
        ):
            assert orig.event_uid == loaded_ev.event_uid
            assert orig.canonical_name == loaded_ev.canonical_name
            assert orig.source_table == loaded_ev.source_table
            if orig.value_num is not None:
                assert loaded_ev.value_num is not None
                assert abs(orig.value_num - loaded_ev.value_num) < 1e-9

    def test_filter_by_stay_ids(self, sample_timeline: list[CanonicalEvent], tmp_path: Path):
        ev2 = _make_event(stay_id=30000002, event_uid="fake-uid-002")
        timelines = {30000001: sample_timeline, 30000002: [ev2]}
        write_timeline_parquet(timelines, tmp_path)
        loaded = read_timeline_parquet(tmp_path, stay_ids=[30000002])
        assert 30000002 in loaded
        assert 30000001 not in loaded

    def test_events_to_dataframe_empty(self):
        df = events_to_dataframe([])
        assert len(df) == 0
        assert "event_uid" in df.columns

    def test_dataframe_to_events_preserves_interval(self, tmp_path: Path):
        ev = _make_event(event_time_end=T3)
        timelines = {30000001: [ev]}
        write_timeline_parquet(timelines, tmp_path)
        loaded = read_timeline_parquet(tmp_path)
        loaded_ev = loaded[30000001][0]
        assert loaded_ev.event_time_end is not None

    def test_partitioned_write(self, sample_timeline: list[CanonicalEvent], tmp_path: Path):
        timelines = {30000001: sample_timeline}
        write_timeline_parquet(timelines, tmp_path, partition_cols=["stay_id"])
        loaded = read_timeline_parquet(tmp_path)
        assert 30000001 in loaded


# ---------------------------------------------------------------------------
# Mixed source multi-table build test
# ---------------------------------------------------------------------------

class TestMultiTableBuild:
    def test_merged_timeline_is_sorted(self, normalizer: Normalizer):
        chart_df = pd.DataFrame({
            "subject_id": [10, 10],
            "hadm_id": [20, 20],
            "stay_id": [30, 30],
            "charttime": pd.to_datetime(["2150-03-01 08:00", "2150-03-01 10:00"]),
            "itemid": [220052, 220052],
            "valuenum": [65.0, 58.0],
            "valueuom": ["mmHg", "mmHg"],
        })
        lab_df = pd.DataFrame({
            "subject_id": [10],
            "hadm_id": [20],
            "charttime": pd.to_datetime(["2150-03-01 09:00"]),
            "itemid": [50971],
            "valuenum": [6.5],
            "valueuom": ["mmol/L"],
        })
        input_df = pd.DataFrame({
            "subject_id": [10],
            "hadm_id": [20],
            "stay_id": [30],
            "starttime": pd.to_datetime(["2150-03-01 09:30"]),
            "endtime": pd.to_datetime(["2150-03-01 11:30"]),
            "itemid": [221906],
            "amount": [2.0],
            "amountuom": ["mg"],
            "rate": [0.1],
            "rateuom": ["mcg/kg/min"],
            "ordercategorydescription": ["Continuous Med"],
            "ordercategoryname": ["Continuous Med"],
            "orderid": [500],
            "linkorderid": [500],
        })
        lookup = {(10, 20): 30}
        tl = build_stay_timeline(
            30,
            chartevents=chart_df,
            labevents=lab_df,
            inputevents=input_df,
            normalizer=normalizer,
            stay_lookup=lookup,
        )
        assert len(tl) == 4
        times = [e.event_time for e in tl]
        assert times == sorted(times), "Timeline is not in chronological order"
        tables = {e.source_table for e in tl}
        assert tables == {"chartevents", "labevents", "inputevents"}
