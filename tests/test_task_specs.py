"""Tests for Phase 2: task-spec YAML loading and pydantic validation."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from mimic_triggerbench.labeling import (
    TaskSpec,
    load_all_task_specs,
    load_task_spec,
)
from mimic_triggerbench.labeling.task_spec_loader import list_available_specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TASK_NAMES = ("hyperkalemia", "hypoglycemia", "hypotension")


def _raw_yaml(task_name: str) -> dict:
    specs_dir = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "mimic_triggerbench"
        / "labeling"
        / "task_specs"
    )
    path = sorted(specs_dir.glob(f"{task_name}_v*.yaml"))[-1]
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Happy-path: all three specs load and validate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", TASK_NAMES)
def test_load_task_spec_valid(task: str) -> None:
    spec = load_task_spec(task)
    assert isinstance(spec, TaskSpec)
    assert spec.task_name == task
    assert spec.spec_version == "v0.1"


def test_load_all_task_specs() -> None:
    specs = load_all_task_specs()
    assert set(specs.keys()) == set(TASK_NAMES)
    for name, spec in specs.items():
        assert spec.task_name == name


def test_list_available_specs() -> None:
    names = list_available_specs()
    assert len(names) >= 3
    for task in TASK_NAMES:
        assert any(task in n for n in names)


# ---------------------------------------------------------------------------
# Specific field checks
# ---------------------------------------------------------------------------


def test_hyperkalemia_threshold() -> None:
    spec = load_task_spec("hyperkalemia")
    assert spec.trigger.threshold == 6.0
    assert spec.trigger.operator == ">="
    assert spec.trigger.unit == "mmol/L"
    assert spec.trigger.sustained_minutes is None
    assert spec.action_window_hours == 6.0


def test_hypoglycemia_threshold() -> None:
    spec = load_task_spec("hypoglycemia")
    assert spec.trigger.threshold == 54.0
    assert spec.trigger.operator == "<"
    assert spec.trigger.unit == "mg/dL"
    assert spec.action_window_hours == 1.0


def test_hypotension_sustained() -> None:
    spec = load_task_spec("hypotension")
    assert spec.trigger.threshold == 65.0
    assert spec.trigger.operator == "<"
    assert spec.trigger.sustained_minutes == 15
    assert spec.clustering is not None
    assert spec.clustering.washout_minutes == 60
    assert spec.action_window_hours == 1.0


def test_negative_windows_present() -> None:
    for task in TASK_NAMES:
        spec = load_task_spec(task)
        assert spec.negative_window.trigger_false_prior_hours == 2.0
        assert spec.negative_window.trigger_false_subsequent_hours == 6.0


def test_all_specs_have_at_least_one_primary_action() -> None:
    for task in TASK_NAMES:
        spec = load_task_spec(task)
        assert any(a.is_primary for a in spec.action_families)


def test_exclusions_present() -> None:
    hk = load_task_spec("hyperkalemia")
    assert len(hk.exclusions) >= 2
    assert any("dialysis" in e.name for e in hk.exclusions)

    hp = load_task_spec("hypotension")
    assert any("comfort" in e.name for e in hp.exclusions)
    assert any("mechanical" in e.name or "circulatory" in e.name for e in hp.exclusions)


# ---------------------------------------------------------------------------
# Rejection tests: malformed specs must fail validation
# ---------------------------------------------------------------------------


def test_reject_missing_trigger() -> None:
    raw = _raw_yaml("hyperkalemia")
    del raw["trigger"]
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(raw)


def test_reject_bad_version_format() -> None:
    raw = _raw_yaml("hyperkalemia")
    raw["spec_version"] = "0.1"
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(raw)


def test_reject_no_action_families() -> None:
    raw = _raw_yaml("hypoglycemia")
    raw["action_families"] = []
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(raw)


def test_reject_all_secondary_actions() -> None:
    raw = _raw_yaml("hypoglycemia")
    for af in raw["action_families"]:
        af["is_primary"] = False
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(raw)


def test_reject_zero_action_window() -> None:
    raw = _raw_yaml("hypotension")
    raw["action_window_hours"] = 0
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(raw)


def test_reject_negative_lookback() -> None:
    raw = _raw_yaml("hyperkalemia")
    raw["evidence_types"][0]["lookback_hours"] = -1
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(raw)


def test_reject_missing_negative_window() -> None:
    raw = _raw_yaml("hypotension")
    del raw["negative_window"]
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(raw)


def test_reject_unknown_task_name() -> None:
    with pytest.raises(FileNotFoundError):
        load_task_spec("sepsis")
