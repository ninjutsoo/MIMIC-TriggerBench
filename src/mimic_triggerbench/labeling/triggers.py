"""Deterministic trigger evaluators for each v1 task family (Phase 5).

Every evaluator uses **as-of slicing only**: the trigger condition at time
``t`` is computed using exclusively events with ``event_time <= t``.

Supported tasks
---------------
- ``hyperkalemia``: latest potassium >= threshold
- ``hypoglycemia``: latest glucose < threshold
- ``hypotension``: MAP < threshold sustained for ``sustained_minutes``
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from mimic_triggerbench.labeling.task_spec_models import TaskSpec, TriggerOperator
from mimic_triggerbench.timeline.models import CanonicalEvent
from mimic_triggerbench.timeline.slicing import get_latest_value, get_recent_values


@dataclass(frozen=True)
class TriggerResult:
    """Outcome of evaluating a trigger at a given time."""

    fired: bool
    signal_value: Optional[float]
    decision_time: datetime
    task_name: str


# ---------------------------------------------------------------------------
# Operator comparison helper
# ---------------------------------------------------------------------------

def _compare(value: float, op: TriggerOperator, threshold: float) -> bool:
    if op == TriggerOperator.GTE:
        return value >= threshold
    if op == TriggerOperator.LTE:
        return value <= threshold
    if op == TriggerOperator.GT:
        return value > threshold
    if op == TriggerOperator.LT:
        return value < threshold
    if op == TriggerOperator.EQ:
        return value == threshold
    raise ValueError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# Lab-based trigger (hyperkalemia / hypoglycemia)
# ---------------------------------------------------------------------------

def evaluate_lab_trigger(
    spec: TaskSpec,
    events: Sequence[CanonicalEvent],
    t: datetime,
) -> TriggerResult:
    """Check a simple lab-value trigger (latest value as-of *t* vs threshold)."""
    latest = get_latest_value(events, spec.trigger.signal, t)
    if latest is None or latest.value_num is None:
        return TriggerResult(fired=False, signal_value=None, decision_time=t, task_name=spec.task_name)

    fired = _compare(latest.value_num, spec.trigger.operator, spec.trigger.threshold)
    return TriggerResult(
        fired=fired,
        signal_value=latest.value_num,
        decision_time=t,
        task_name=spec.task_name,
    )


# ---------------------------------------------------------------------------
# Sustained-condition trigger (hypotension)
# ---------------------------------------------------------------------------

def evaluate_sustained_trigger(
    spec: TaskSpec,
    events: Sequence[CanonicalEvent],
    t: datetime,
) -> TriggerResult:
    """Check a sustained-condition trigger (signal below/above threshold for N minutes).

    For MAP < 65 sustained 15 min: gather all signal readings up to ``t``
    that satisfy the trigger condition in an unbroken tail (no intervening
    reading that violates the condition).  If the span of that unbroken
    tail is >= ``sustained_minutes``, the trigger fires.
    """
    sustained_min = spec.trigger.sustained_minutes
    if sustained_min is None:
        return evaluate_lab_trigger(spec, events, t)

    all_signal = [
        ev for ev in events
        if ev.canonical_name == spec.trigger.signal
        and ev.event_time <= t
        and ev.value_num is not None
    ]
    all_signal.sort(key=lambda ev: ev.event_time)

    if not all_signal:
        return TriggerResult(fired=False, signal_value=None, decision_time=t, task_name=spec.task_name)

    latest_value = all_signal[-1].value_num

    tail_start: Optional[datetime] = None
    for ev in reversed(all_signal):
        if ev.value_num is not None and _compare(ev.value_num, spec.trigger.operator, spec.trigger.threshold):
            tail_start = ev.event_time
        else:
            break

    if tail_start is None:
        return TriggerResult(fired=False, signal_value=latest_value, decision_time=t, task_name=spec.task_name)

    span_min = (t - tail_start).total_seconds() / 60.0
    fired = span_min >= sustained_min

    return TriggerResult(
        fired=fired,
        signal_value=latest_value,
        decision_time=t,
        task_name=spec.task_name,
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_SUSTAINED_TASKS = {"hypotension"}


def evaluate_trigger(
    spec: TaskSpec,
    events: Sequence[CanonicalEvent],
    t: datetime,
) -> TriggerResult:
    """Evaluate the trigger for *spec* at time *t* on *events*."""
    if spec.task_name in _SUSTAINED_TASKS or spec.trigger.sustained_minutes is not None:
        return evaluate_sustained_trigger(spec, events, t)
    return evaluate_lab_trigger(spec, events, t)
