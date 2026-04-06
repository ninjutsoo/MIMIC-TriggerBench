"""Deterministic episode generators for the three v1 tasks (Phase 5).

Each generator walks a stay's canonical timeline, fires triggers, and
produces :class:`Episode` rows with:

- **accepted_action_families**: protocol-derived gold actions from the task spec
- **observed_action_families**: what clinicians actually did (detected from events
  *after* ``decision_time`` within the action window)

Negative-window future-looking logic (the ``trigger_false_subsequent_hours``
requirement) is used **only** for deciding which candidate decision times
qualify as negatives.  It is never stored on the episode, never exposed to
replay tools, and never accessible to the agent.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Sequence

from mimic_triggerbench.feasibility.detectors import (
    detect_dialysis_start,
    detect_fluid_bolus,
    detect_insulin_dextrose_pairing,
    detect_vasopressor_actions,
)
from mimic_triggerbench.labeling.episode_models import Episode, generate_episode_id
from mimic_triggerbench.labeling.task_spec_models import TaskSpec
from mimic_triggerbench.labeling.triggers import TriggerResult, evaluate_trigger
from mimic_triggerbench.timeline.models import CanonicalEvent
from mimic_triggerbench.timeline.slicing import slice_as_of, slice_window

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observed-action detection helpers
# ---------------------------------------------------------------------------

def _detect_observed_actions_in_window(
    events: Sequence[CanonicalEvent],
    window_start: datetime,
    window_end: datetime,
) -> set[str]:
    """Run all action-family detectors on events within [window_start, window_end].

    Returns the set of observed action family names.
    """
    window_events = slice_window(list(events), window_start, window_end)
    if not window_events:
        return set()

    families: set[str] = set()

    for det in detect_insulin_dextrose_pairing(window_events):
        families.add(det.action_family)
    for det in detect_vasopressor_actions(window_events):
        families.add(det.action_family)
    for det in detect_fluid_bolus(window_events):
        families.add(det.action_family)
    for det in detect_dialysis_start(window_events):
        families.add(det.action_family)

    return families


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def _accepted_families(spec: TaskSpec) -> list[str]:
    """Primary action family names from the task spec (protocol-derived gold)."""
    return sorted(af.name for af in spec.action_families if af.is_primary)


def _mandatory_evidence(spec: TaskSpec) -> list[str]:
    return sorted(et.name for et in spec.evidence_types if et.required)


def _context_start(
    events: Sequence[CanonicalEvent],
    decision_time: datetime,
) -> datetime:
    """Earliest event_time in the timeline at or before decision_time."""
    as_of_events = slice_as_of(list(events), decision_time)
    if as_of_events:
        return min(e.event_time for e in as_of_events)
    return decision_time


# ---------------------------------------------------------------------------
# Candidate signal times
# ---------------------------------------------------------------------------

def _signal_times(
    events: Sequence[CanonicalEvent],
    signal_name: str,
) -> list[tuple[datetime, Optional[float]]]:
    """Return sorted (event_time, value_num) pairs for the given signal."""
    pairs: list[tuple[datetime, Optional[float]]] = []
    for e in events:
        if e.canonical_name == signal_name:
            pairs.append((e.event_time, e.value_num))
    pairs.sort(key=lambda p: p[0])
    return pairs


# ---------------------------------------------------------------------------
# Positive episode generation
# ---------------------------------------------------------------------------

def _generate_positive_episodes_lab(
    spec: TaskSpec,
    events: list[CanonicalEvent],
    stay_id: int,
    subject_id: int,
    hadm_id: int,
) -> list[Episode]:
    """Generate positive episodes for a lab-based trigger (hyperkalemia / hypoglycemia)."""
    signal_times = _signal_times(events, spec.trigger.signal)
    episodes: list[Episode] = []

    for t, _ in signal_times:
        result = evaluate_trigger(spec, events, t)
        if not result.fired:
            continue

        eid = generate_episode_id(spec.task_name, stay_id, t, "positive")
        action_window_end = t + timedelta(hours=spec.action_window_hours)
        observed = _detect_observed_actions_in_window(events, t, action_window_end)

        episodes.append(Episode(
            episode_id=eid,
            task_name=spec.task_name,
            subject_id=subject_id,
            hadm_id=hadm_id,
            stay_id=stay_id,
            decision_time=t,
            context_start=_context_start(events, t),
            trigger_label=True,
            trigger_type="positive",
            trigger_value=result.signal_value,
            accepted_action_families=_accepted_families(spec),
            observed_action_families=sorted(observed),
            mandatory_evidence_types=_mandatory_evidence(spec),
        ))

    return episodes


def _generate_positive_episodes_sustained(
    spec: TaskSpec,
    events: list[CanonicalEvent],
    stay_id: int,
    subject_id: int,
    hadm_id: int,
) -> list[Episode]:
    """Generate positive episodes for a sustained trigger (hypotension).

    The trigger fires at the earliest time ``t`` where the sustained
    condition is met.  After a trigger fires, a washout period applies
    before the next episode can fire (per clustering rules).
    """
    signal_times = _signal_times(events, spec.trigger.signal)
    episodes: list[Episode] = []
    last_fire_time: Optional[datetime] = None
    washout_min = spec.clustering.washout_minutes if spec.clustering else 0

    for t, _ in signal_times:
        if last_fire_time is not None:
            if washout_min > 0:
                gap_events = slice_window(
                    events, last_fire_time, t,
                )
                signal_in_gap = [
                    e for e in gap_events
                    if e.canonical_name == spec.trigger.signal
                    and e.value_num is not None
                    and e.value_num >= spec.trigger.threshold
                ]
                if not signal_in_gap:
                    continue
                earliest_recovery = min(e.event_time for e in signal_in_gap)
                if (t - earliest_recovery).total_seconds() / 60.0 < washout_min:
                    continue

        result = evaluate_trigger(spec, events, t)
        if not result.fired:
            continue

        eid = generate_episode_id(spec.task_name, stay_id, t, "positive")
        action_window_end = t + timedelta(hours=spec.action_window_hours)
        observed = _detect_observed_actions_in_window(events, t, action_window_end)

        episodes.append(Episode(
            episode_id=eid,
            task_name=spec.task_name,
            subject_id=subject_id,
            hadm_id=hadm_id,
            stay_id=stay_id,
            decision_time=t,
            context_start=_context_start(events, t),
            trigger_label=True,
            trigger_type="positive",
            trigger_value=result.signal_value,
            accepted_action_families=_accepted_families(spec),
            observed_action_families=sorted(observed),
            mandatory_evidence_types=_mandatory_evidence(spec),
        ))
        last_fire_time = t

    return episodes


# ---------------------------------------------------------------------------
# Negative episode generation
# ---------------------------------------------------------------------------

def _generate_negative_episodes(
    spec: TaskSpec,
    events: list[CanonicalEvent],
    stay_id: int,
    subject_id: int,
    hadm_id: int,
    positive_times: set[datetime],
) -> list[Episode]:
    """Generate negative episodes for a single stay.

    A candidate time ``t`` qualifies as a negative episode when:

    1. The trigger is **false** at ``t``.
    2. The trigger has been false for the prior ``trigger_false_prior_hours``.
    3. The trigger **remains false** for the subsequent
       ``trigger_false_subsequent_hours`` (future-looking, label-generation-only).

    Condition 3 uses future data but this is strictly internal:
    the resulting episode carries no future-derived fields.
    """
    neg_def = spec.negative_window
    prior_hours = neg_def.trigger_false_prior_hours
    subsequent_hours = neg_def.trigger_false_subsequent_hours

    signal_times = _signal_times(events, spec.trigger.signal)
    if not signal_times:
        return []

    episodes: list[Episode] = []
    seen_decision_times: set[datetime] = set()

    for t, _ in signal_times:
        if t in positive_times or t in seen_decision_times:
            continue

        result = evaluate_trigger(spec, events, t)
        if result.fired:
            continue

        # Condition 2: trigger false for prior N hours
        prior_start = t - timedelta(hours=prior_hours)
        prior_signals = [
            (st, val) for st, val in signal_times
            if prior_start <= st < t and val is not None
        ]
        if any(_trigger_condition_met(spec, val) for _, val in prior_signals):
            continue

        # Condition 3: trigger remains false for subsequent N hours (FUTURE-LOOKING)
        subsequent_end = t + timedelta(hours=subsequent_hours)
        subsequent_signals = [
            (st, val) for st, val in signal_times
            if t < st <= subsequent_end and val is not None
        ]
        if any(_trigger_condition_met(spec, val) for _, val in subsequent_signals):
            continue

        seen_decision_times.add(t)
        eid = generate_episode_id(spec.task_name, stay_id, t, "negative")

        episodes.append(Episode(
            episode_id=eid,
            task_name=spec.task_name,
            subject_id=subject_id,
            hadm_id=hadm_id,
            stay_id=stay_id,
            decision_time=t,
            context_start=_context_start(events, t),
            trigger_label=False,
            trigger_type="negative",
            trigger_value=result.signal_value,
            accepted_action_families=[],
            observed_action_families=[],
            mandatory_evidence_types=_mandatory_evidence(spec),
        ))

    return episodes


def _trigger_condition_met(spec: TaskSpec, value: float) -> bool:
    """Check whether a raw signal value satisfies the trigger threshold."""
    from mimic_triggerbench.labeling.triggers import _compare
    return _compare(value, spec.trigger.operator, spec.trigger.threshold)


# ---------------------------------------------------------------------------
# Per-stay episode generation
# ---------------------------------------------------------------------------

_SUSTAINED_TASKS = {"hypotension"}


def generate_episodes_for_stay(
    spec: TaskSpec,
    events: list[CanonicalEvent],
    stay_id: int,
    subject_id: int,
    hadm_id: int,
) -> list[Episode]:
    """Generate all episodes (positive + negative) for one stay and one task."""
    if spec.task_name in _SUSTAINED_TASKS or spec.trigger.sustained_minutes is not None:
        positives = _generate_positive_episodes_sustained(
            spec, events, stay_id, subject_id, hadm_id,
        )
    else:
        positives = _generate_positive_episodes_lab(
            spec, events, stay_id, subject_id, hadm_id,
        )

    positive_times = {ep.decision_time for ep in positives}
    negatives = _generate_negative_episodes(
        spec, events, stay_id, subject_id, hadm_id, positive_times,
    )

    all_episodes = positives + negatives
    all_episodes.sort(key=lambda ep: (ep.decision_time, ep.trigger_type))
    return all_episodes


# ---------------------------------------------------------------------------
# Bulk generation across stays
# ---------------------------------------------------------------------------

def generate_all_episodes(
    spec: TaskSpec,
    timelines: dict[int, list[CanonicalEvent]],
    *,
    icustays_metadata: Optional[dict[int, tuple[int, int]]] = None,
) -> list[Episode]:
    """Generate episodes for all stays in *timelines* for one task.

    *icustays_metadata* maps ``stay_id → (subject_id, hadm_id)``.
    If not provided, subject_id/hadm_id are taken from the first event.
    """
    all_episodes: list[Episode] = []

    for stay_id in sorted(timelines.keys()):
        events = timelines[stay_id]
        if not events:
            continue

        if icustays_metadata and stay_id in icustays_metadata:
            subject_id, hadm_id = icustays_metadata[stay_id]
        else:
            subject_id = events[0].subject_id
            hadm_id = events[0].hadm_id

        stay_episodes = generate_episodes_for_stay(
            spec, events, stay_id, subject_id, hadm_id,
        )
        all_episodes.extend(stay_episodes)

    logger.info(
        "Task %s: generated %d episodes (%d positive, %d negative) across %d stays",
        spec.task_name,
        len(all_episodes),
        sum(1 for e in all_episodes if e.trigger_label),
        sum(1 for e in all_episodes if not e.trigger_label),
        len(timelines),
    )
    return all_episodes


# ---------------------------------------------------------------------------
# Episode I/O
# ---------------------------------------------------------------------------

def episodes_to_records(episodes: list[Episode]) -> list[dict]:
    """Convert episodes to flat dicts suitable for DataFrame / JSON serialization."""
    records = []
    for ep in episodes:
        records.append({
            "episode_id": ep.episode_id,
            "task_name": ep.task_name,
            "subject_id": ep.subject_id,
            "hadm_id": ep.hadm_id,
            "stay_id": ep.stay_id,
            "decision_time": ep.decision_time.isoformat(),
            "context_start": ep.context_start.isoformat(),
            "trigger_label": ep.trigger_label,
            "trigger_type": ep.trigger_type,
            "trigger_value": ep.trigger_value,
            "accepted_action_families": ep.accepted_action_families,
            "observed_action_families": ep.observed_action_families,
            "mandatory_evidence_types": ep.mandatory_evidence_types,
            "split": ep.split,
        })
    return records
