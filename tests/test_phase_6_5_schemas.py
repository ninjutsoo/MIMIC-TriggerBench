from __future__ import annotations

import copy

import pytest

from mimic_triggerbench.schemas import (
    OutputSchemaError,
    validate_benchmark_output,
    validate_episode_input,
    validate_tool_result,
    IntermediateAgentState,
)


# ---- BenchmarkOutput fixtures / tests (existing) --------------------------


def _valid_payload() -> dict:
    return {
        "episode_id": "ep_123",
        "task_name": "hyperkalemia",
        "trigger_detected": True,
        "trigger_type": "positive",
        "decision_time": "2150-03-01T08:00:00",
        "urgency_level": "high",
        "recommended_next_steps": ["administer calcium", "consider insulin+dextrose"],
        "recommended_action_families": ["calcium", "insulin_dextrose"],
        "evidence": [
            {
                "source_table": "labevents",
                "canonical_name": "potassium",
                "event_time": "2150-03-01T07:55:00",
                "value": 6.7,
                "why_relevant": "K is severely elevated at trigger time.",
            }
        ],
        "missing_information": ["ekg"],
        "abstain": False,
        "abstain_reason": None,
        "confidence": 0.85,
        "tool_trace": [
            {
                "tool_name": "get_recent_labs",
                "arguments": {"lab_names": ["potassium"], "hours_back": 24},
                "returned_count": 3,
            }
        ],
    }


def test_valid_payload_parses() -> None:
    out = validate_benchmark_output(_valid_payload())
    assert out.task_name == "hyperkalemia"
    assert out.urgency_level == "high"
    assert out.decision_time.isoformat().startswith("2150-03-01T08:00:00")
    assert out.evidence[0].canonical_name == "potassium"


def test_missing_required_field_fails() -> None:
    payload = _valid_payload()
    payload.pop("episode_id")
    with pytest.raises(OutputSchemaError):
        validate_benchmark_output(payload)


def test_extra_field_forbidden_fails() -> None:
    payload = _valid_payload()
    payload["unexpected"] = "nope"
    with pytest.raises(OutputSchemaError):
        validate_benchmark_output(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task_name", "sepsis"),
        ("urgency_level", "urgent"),
    ],
)
def test_enum_constraints_fail(field: str, value: str) -> None:
    payload = _valid_payload()
    payload[field] = value
    with pytest.raises(OutputSchemaError):
        validate_benchmark_output(payload)


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 2.0])
def test_confidence_bounds_enforced(confidence: float) -> None:
    payload = _valid_payload()
    payload["confidence"] = confidence
    with pytest.raises(OutputSchemaError):
        validate_benchmark_output(payload)


def test_tool_trace_returned_count_nonnegative() -> None:
    payload = _valid_payload()
    payload["tool_trace"][0]["returned_count"] = -1
    with pytest.raises(OutputSchemaError):
        validate_benchmark_output(payload)


def test_evidence_value_may_be_string() -> None:
    payload = _valid_payload()
    payload["evidence"][0]["value"] = "6.7 mmol/L"
    out = validate_benchmark_output(payload)
    assert isinstance(out.evidence[0].value, str)


def test_nested_extra_field_forbidden() -> None:
    payload = _valid_payload()
    payload2 = copy.deepcopy(payload)
    payload2["evidence"][0]["extra"] = 1
    with pytest.raises(OutputSchemaError):
        validate_benchmark_output(payload2)


# ---- EpisodeInput tests ---------------------------------------------------


def _valid_episode_input() -> dict:
    return {
        "episode_id": "ep_001",
        "task_name": "hypoglycemia",
        "stay_id": 30000001,
        "hadm_id": 20000001,
        "subject_id": 10000001,
        "decision_time": "2150-03-01T10:00:00",
        "context_start": "2150-02-28T22:00:00",
        "mandatory_evidence_types": ["glucose_trend", "insulin_recent"],
        "accepted_action_families": ["iv_dextrose", "glucagon"],
    }


def test_episode_input_valid() -> None:
    inp = validate_episode_input(_valid_episode_input())
    assert inp.task_name == "hypoglycemia"
    assert inp.stay_id == 30000001


def test_episode_input_missing_field() -> None:
    payload = _valid_episode_input()
    payload.pop("decision_time")
    with pytest.raises(OutputSchemaError):
        validate_episode_input(payload)


def test_episode_input_extra_field_forbidden() -> None:
    payload = _valid_episode_input()
    payload["extra_field"] = True
    with pytest.raises(OutputSchemaError):
        validate_episode_input(payload)


def test_episode_input_bad_task_name() -> None:
    payload = _valid_episode_input()
    payload["task_name"] = "sepsis"
    with pytest.raises(OutputSchemaError):
        validate_episode_input(payload)


def test_episode_input_defaults() -> None:
    payload = {
        "episode_id": "ep_002",
        "task_name": "hypotension",
        "stay_id": 1,
        "hadm_id": 1,
        "subject_id": 1,
        "decision_time": "2150-01-01T00:00:00",
        "context_start": "2150-01-01T00:00:00",
    }
    inp = validate_episode_input(payload)
    assert inp.mandatory_evidence_types == []
    assert inp.accepted_action_families == []


# ---- ToolResult tests ------------------------------------------------------


def _valid_tool_result() -> dict:
    return {
        "tool_name": "get_recent_labs",
        "queried_time_range": {
            "start": "2150-03-01T00:00:00",
            "end": "2150-03-01T08:00:00",
        },
        "provenance": {
            "source_tables": ["labevents"],
            "mapping_version": "v0.1",
        },
        "results": [
            {
                "canonical_name": "potassium",
                "event_time": "2150-03-01T07:55:00",
                "value": 6.7,
                "unit": "mmol/L",
                "source_table": "labevents",
                "event_uid": "abcd-1234",
                "metadata": None,
            }
        ],
        "result_count": 1,
    }


def test_tool_result_valid() -> None:
    tr = validate_tool_result(_valid_tool_result())
    assert tr.tool_name == "get_recent_labs"
    assert tr.result_count == 1
    assert tr.results[0].canonical_name == "potassium"
    assert tr.queried_time_range.start.year == 2150


def test_tool_result_missing_provenance() -> None:
    payload = _valid_tool_result()
    payload.pop("provenance")
    with pytest.raises(OutputSchemaError):
        validate_tool_result(payload)


def test_tool_result_extra_field_on_row() -> None:
    payload = _valid_tool_result()
    payload["results"][0]["bonus"] = "bad"
    with pytest.raises(OutputSchemaError):
        validate_tool_result(payload)


def test_tool_result_extra_field_on_envelope() -> None:
    payload = _valid_tool_result()
    payload["extra"] = "bad"
    with pytest.raises(OutputSchemaError):
        validate_tool_result(payload)


def test_tool_result_negative_count_rejected() -> None:
    payload = _valid_tool_result()
    payload["result_count"] = -1
    with pytest.raises(OutputSchemaError):
        validate_tool_result(payload)


def test_tool_result_row_value_nullable() -> None:
    payload = _valid_tool_result()
    payload["results"][0]["value"] = None
    tr = validate_tool_result(payload)
    assert tr.results[0].value is None


def test_tool_result_extra_on_time_range() -> None:
    payload = _valid_tool_result()
    payload["queried_time_range"]["zone"] = "UTC"
    with pytest.raises(OutputSchemaError):
        validate_tool_result(payload)


def test_tool_result_extra_on_provenance() -> None:
    payload = _valid_tool_result()
    payload["provenance"]["extra"] = True
    with pytest.raises(OutputSchemaError):
        validate_tool_result(payload)


# ---- IntermediateAgentState tests ------------------------------------------


def test_intermediate_state_round_trip() -> None:
    state = IntermediateAgentState(
        episode_id="ep_001",
        task_name="hyperkalemia",
        decision_time="2150-03-01T08:00:00",
        tool_calls=[],
        gathered_evidence=[],
        reasoning_notes=["initial assessment"],
        iteration=0,
    )
    dumped = state.model_dump(mode="json")
    restored = IntermediateAgentState.model_validate(dumped)
    assert restored.episode_id == "ep_001"
    assert restored.reasoning_notes == ["initial assessment"]


def test_intermediate_state_extra_forbidden() -> None:
    with pytest.raises(Exception):
        IntermediateAgentState(
            episode_id="ep_001",
            task_name="hyperkalemia",
            decision_time="2150-03-01T08:00:00",
            iteration=0,
            secret="leak",
        )

