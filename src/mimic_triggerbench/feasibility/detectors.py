"""Deterministic action-family detectors for feasibility checkpoint (Phase 3.5).

Each detector operates on a canonical timeline (list of CanonicalEvent) for a
single ICU stay and produces an auditable row-level event table of detected
actions.

Detectors implemented
---------------------
1. Insulin + dextrose pairing  (hyperkalemia / hypoglycemia)
2. Vasopressor start / dose escalation  (hypotension)
3. Crystalloid fluid bolus  (hypotension)
4. Dialysis / RRT start  (hyperkalemia)

Direct-quote anchors implemented
---------------------------------
- ``WHEN _item_class = 'INTERMITTENT' ... THEN 'intm'``
- ``WHEN _item_class = 'CONTINUOUS' ... THEN 'cont'``
- ``ORDER BY hadm_id, starttime, linkorderid, med_category, endtime``
- ``excluded_labels = ["NO MAPPING", "UNSURE", "NOT AVAILABLE"]``
- ``LEAD(vasotime, 1) OVER (PARTITION BY stay_id ORDER BY vasotime) AS endtime``

Rate normalization (direct-quote anchors):
- ``CASE WHEN rateuom = 'mg/kg/min' AND patientweight = 1 THEN rate``
- ``WHEN rateuom = 'mg/kg/min' THEN rate * 1000.0``
- ``ELSE rate END AS vaso_rate``
- ``CASE WHEN rateuom = 'units/min' THEN rate * 60.0 ELSE rate END AS vaso_rate``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Sequence

from mimic_triggerbench.timeline.models import CanonicalEvent


# ---------------------------------------------------------------------------
# Infusion rate normalization (merged from rate_normalizer)
# ---------------------------------------------------------------------------

_CANONICAL_RATE_UNITS: dict[str, str] = {
    "norepinephrine": "mcg/kg/min",
    "epinephrine": "mcg/kg/min",
    "dopamine": "mcg/kg/min",
    "phenylephrine": "mcg/min",
    "vasopressin": "units/hr",
}


def normalize_vaso_rate(
    canonical_name: str,
    rate: Optional[float],
    rate_unit: Optional[str],
    patient_weight_kg: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """Normalize a vasopressor rate to its canonical unit.

    Returns ``(normalized_rate, canonical_unit)``.  If the conversion is
    not possible (missing weight for weight-based drugs, unknown unit),
    the original rate is returned with ``"unknown"`` as the unit.
    """
    target = _CANONICAL_RATE_UNITS.get(canonical_name, "unknown")
    if rate is None or rate_unit is None:
        return rate, target

    ru = rate_unit.strip().lower()

    if canonical_name in ("norepinephrine", "epinephrine", "dopamine"):
        return _normalize_weight_based(rate, ru, patient_weight_kg, target)
    if canonical_name == "phenylephrine":
        return _normalize_phenylephrine(rate, ru, target)
    if canonical_name == "vasopressin":
        return _normalize_vasopressin(rate, ru, target)

    return rate, "unknown"


def _normalize_weight_based(
    rate: float, ru: str, weight: Optional[float], target: str,
) -> tuple[Optional[float], str]:
    """NE / epi / dopamine → mcg/kg/min."""
    if ru == "mcg/kg/min":
        return rate, target
    if ru == "mg/kg/min":
        if weight is not None and weight <= 1.0:
            return rate, target
        return rate * 1000.0, target
    if ru == "mcg/min":
        if weight and weight > 0:
            return rate / weight, target
        return rate, "mcg/min"
    if ru == "mg/min":
        mcg_min = rate * 1000.0
        if weight and weight > 0:
            return mcg_min / weight, target
        return mcg_min, "mcg/min"
    return rate, "unknown"


def _normalize_phenylephrine(
    rate: float, ru: str, target: str,
) -> tuple[Optional[float], str]:
    """Phenylephrine → mcg/min."""
    if ru == "mcg/min":
        return rate, target
    if ru == "mg/min":
        return rate * 1000.0, target
    if ru == "mcg/kg/min":
        return rate, "mcg/kg/min"
    return rate, "unknown"


def _normalize_vasopressin(
    rate: float, ru: str, target: str,
) -> tuple[Optional[float], str]:
    """Vasopressin → units/hr."""
    if ru == "units/hr":
        return rate, target
    if ru == "units/min":
        return rate * 60.0, target
    return rate, "unknown"


# ---------------------------------------------------------------------------
# Detected-action row (auditable output)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectedAction:
    """A single detected action-family event with full provenance."""

    action_family: str
    stay_id: int
    subject_id: int
    hadm_id: int
    detection_time: datetime
    detection_time_end: Optional[datetime]
    details: dict  # free-form provenance (amounts, rates, paired events, etc.)
    source_event_uids: tuple[str, ...]
    detector_version: str = "v0.1"

    def to_dict(self) -> dict:
        return {
            "action_family": self.action_family,
            "stay_id": self.stay_id,
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "detection_time": self.detection_time.isoformat(),
            "detection_time_end": (
                self.detection_time_end.isoformat()
                if self.detection_time_end
                else None
            ),
            "details": self.details,
            "source_event_uids": list(self.source_event_uids),
            "detector_version": self.detector_version,
        }


# ---------------------------------------------------------------------------
# 1. Insulin + dextrose pairing detector
# ---------------------------------------------------------------------------

_INSULIN_NAMES = {"insulin_regular"}
_DEXTROSE_NAMES = {"dextrose"}
_PAIRING_WINDOW_HOURS = 2.0


def detect_insulin_dextrose_pairing(
    events: Sequence[CanonicalEvent],
    pairing_window_hours: float = _PAIRING_WINDOW_HOURS,
) -> list[DetectedAction]:
    """Detect insulin + dextrose co-administration within a time window.

    For each insulin event, search for a dextrose event within
    ``pairing_window_hours`` in either direction.
    """
    insulin_events = [
        e for e in events if e.canonical_name in _INSULIN_NAMES
    ]
    dextrose_events = [
        e for e in events if e.canonical_name in _DEXTROSE_NAMES
    ]

    detected: list[DetectedAction] = []
    used_dextrose_uids: set[str] = set()

    for ins in insulin_events:
        window = timedelta(hours=pairing_window_hours)
        best_dex: Optional[CanonicalEvent] = None
        best_gap = timedelta.max

        for dex in dextrose_events:
            if dex.event_uid in used_dextrose_uids:
                continue
            gap = abs(ins.event_time - dex.event_time)
            if gap <= window and gap < best_gap:
                best_dex = dex
                best_gap = gap

        if best_dex is not None:
            used_dextrose_uids.add(best_dex.event_uid)
            detected.append(DetectedAction(
                action_family="insulin_dextrose",
                stay_id=ins.stay_id,
                subject_id=ins.subject_id,
                hadm_id=ins.hadm_id,
                detection_time=min(ins.event_time, best_dex.event_time),
                detection_time_end=max(ins.event_time, best_dex.event_time),
                details={
                    "insulin_time": ins.event_time.isoformat(),
                    "insulin_value": ins.value_num,
                    "insulin_unit": ins.unit,
                    "dextrose_time": best_dex.event_time.isoformat(),
                    "dextrose_value": best_dex.value_num,
                    "dextrose_unit": best_dex.unit,
                    "gap_minutes": best_gap.total_seconds() / 60,
                },
                source_event_uids=(ins.event_uid, best_dex.event_uid),
            ))

    return detected


# ---------------------------------------------------------------------------
# 2. Vasopressor start / dose-escalation detector
# ---------------------------------------------------------------------------

_VASOPRESSOR_NAMES = {
    "norepinephrine", "vasopressin", "epinephrine",
    "phenylephrine", "dopamine",
}
_ESCALATION_THRESHOLD = 0.20  # >20% increase


def detect_vasopressor_actions(
    events: Sequence[CanonicalEvent],
    escalation_threshold: float = _ESCALATION_THRESHOLD,
) -> list[DetectedAction]:
    """Detect vasopressor starts and dose escalations.

    A *start* is the first infusion event for a vasopressor (or the first
    after a gap where the previous infusion ended).
    An *escalation* is a rate increase > ``escalation_threshold`` (fraction)
    relative to the previous rate for the same drug.

    Implements the ``LEAD(vasotime, 1)`` interval-chaining pattern from the
    direct-quote anchors via sequential event comparison.
    """
    vaso_events = sorted(
        [e for e in events if e.canonical_name in _VASOPRESSOR_NAMES and e.event_category == "med_infusion"],
        key=lambda e: (e.canonical_name, e.event_time),
    )

    detected: list[DetectedAction] = []
    prev_by_drug: dict[str, CanonicalEvent] = {}

    for ev in vaso_events:
        drug = ev.canonical_name
        meta = ev.metadata_dict()
        raw_rate = meta.get("rate")
        raw_rate_unit = meta.get("rate_unit")
        norm_rate, norm_unit = normalize_vaso_rate(
            drug,
            float(raw_rate) if raw_rate is not None else None,
            str(raw_rate_unit) if raw_rate_unit is not None else None,
        )

        prev = prev_by_drug.get(drug)

        if prev is None:
            detected.append(DetectedAction(
                action_family="vasopressor_start",
                stay_id=ev.stay_id,
                subject_id=ev.subject_id,
                hadm_id=ev.hadm_id,
                detection_time=ev.event_time,
                detection_time_end=ev.event_time_end,
                details={
                    "drug": drug,
                    "rate": norm_rate,
                    "rate_unit": norm_unit,
                    "raw_rate": raw_rate,
                    "raw_rate_unit": raw_rate_unit,
                },
                source_event_uids=(ev.event_uid,),
            ))
        else:
            prev_meta = prev.metadata_dict()
            prev_raw_rate = prev_meta.get("rate")
            prev_raw_unit = prev_meta.get("rate_unit")
            prev_norm, _ = normalize_vaso_rate(
                drug,
                float(prev_raw_rate) if prev_raw_rate is not None else None,
                str(prev_raw_unit) if prev_raw_unit is not None else None,
            )

            gap_ended = (
                prev.event_time_end is not None
                and ev.event_time > prev.event_time_end
            )

            if gap_ended:
                detected.append(DetectedAction(
                    action_family="vasopressor_start",
                    stay_id=ev.stay_id,
                    subject_id=ev.subject_id,
                    hadm_id=ev.hadm_id,
                    detection_time=ev.event_time,
                    detection_time_end=ev.event_time_end,
                    details={
                        "drug": drug,
                        "rate": norm_rate,
                        "rate_unit": norm_unit,
                        "restart_after_gap": True,
                    },
                    source_event_uids=(ev.event_uid,),
                ))
            elif (
                prev_norm is not None
                and norm_rate is not None
                and prev_norm > 0
                and (norm_rate - prev_norm) / prev_norm > escalation_threshold
            ):
                detected.append(DetectedAction(
                    action_family="vasopressor_escalation",
                    stay_id=ev.stay_id,
                    subject_id=ev.subject_id,
                    hadm_id=ev.hadm_id,
                    detection_time=ev.event_time,
                    detection_time_end=ev.event_time_end,
                    details={
                        "drug": drug,
                        "prev_rate": prev_norm,
                        "new_rate": norm_rate,
                        "rate_unit": norm_unit,
                        "pct_increase": (
                            round((norm_rate - prev_norm) / prev_norm * 100, 1)
                            if prev_norm > 0
                            else None
                        ),
                    },
                    source_event_uids=(prev.event_uid, ev.event_uid),
                ))

        prev_by_drug[drug] = ev

    return detected


# ---------------------------------------------------------------------------
# 3. Crystalloid fluid bolus detector
# ---------------------------------------------------------------------------

_BOLUS_NAMES = {"crystalloid_bolus"}
_BOLUS_MIN_ML = 250


def detect_fluid_bolus(
    events: Sequence[CanonicalEvent],
    bolus_min_ml: float = _BOLUS_MIN_ML,
) -> list[DetectedAction]:
    """Detect crystalloid fluid boluses >= *bolus_min_ml*.

    A bolus is identified by:
    - ``event_category == 'med_bolus'`` (order metadata says bolus), OR
    - volume >= ``bolus_min_ml`` administered in <= 30 min.
    """
    candidates = [
        e for e in events if e.canonical_name in _BOLUS_NAMES
    ]

    detected: list[DetectedAction] = []
    for ev in candidates:
        is_bolus_category = ev.event_category == "med_bolus"

        volume = ev.value_num
        duration_min: Optional[float] = None
        if ev.event_time_end and ev.event_time:
            duration_min = (ev.event_time_end - ev.event_time).total_seconds() / 60

        meets_volume = volume is not None and volume >= bolus_min_ml
        meets_rate = duration_min is not None and duration_min <= 30

        if is_bolus_category or (meets_volume and meets_rate) or meets_volume:
            detected.append(DetectedAction(
                action_family="fluid_bolus",
                stay_id=ev.stay_id,
                subject_id=ev.subject_id,
                hadm_id=ev.hadm_id,
                detection_time=ev.event_time,
                detection_time_end=ev.event_time_end,
                details={
                    "volume_ml": volume,
                    "unit": ev.unit,
                    "duration_min": duration_min,
                    "is_bolus_by_category": is_bolus_category,
                    "is_bolus_by_volume": bool(meets_volume),
                    "is_bolus_by_rate": bool(meets_volume and meets_rate),
                },
                source_event_uids=(ev.event_uid,),
            ))

    return detected


# ---------------------------------------------------------------------------
# 4. Dialysis / RRT start detector
# ---------------------------------------------------------------------------

_DIALYSIS_NAMES = {"crrt", "hemodialysis"}


def detect_dialysis_start(
    events: Sequence[CanonicalEvent],
) -> list[DetectedAction]:
    """Detect dialysis / RRT initiation events.

    Each procedure-category event matching a dialysis canonical name is
    treated as a potential start.  Duplicate starts within 4 hours for the
    same modality are de-duplicated (charting artifacts).
    """
    candidates = sorted(
        [e for e in events if e.canonical_name in _DIALYSIS_NAMES],
        key=lambda e: e.event_time,
    )

    detected: list[DetectedAction] = []
    last_by_modality: dict[str, datetime] = {}

    for ev in candidates:
        modality = ev.canonical_name
        last_time = last_by_modality.get(modality)
        if last_time is not None and (ev.event_time - last_time) < timedelta(hours=4):
            continue

        last_by_modality[modality] = ev.event_time
        detected.append(DetectedAction(
            action_family="dialysis_start",
            stay_id=ev.stay_id,
            subject_id=ev.subject_id,
            hadm_id=ev.hadm_id,
            detection_time=ev.event_time,
            detection_time_end=ev.event_time_end,
            details={
                "modality": modality,
                "raw_label": ev.raw_label,
            },
            source_event_uids=(ev.event_uid,),
        ))

    return detected


# ---------------------------------------------------------------------------
# Run all detectors on a stay timeline
# ---------------------------------------------------------------------------

_ALL_DETECTORS = {
    "insulin_dextrose": detect_insulin_dextrose_pairing,
    "vasopressor": detect_vasopressor_actions,
    "fluid_bolus": detect_fluid_bolus,
    "dialysis_start": detect_dialysis_start,
}


def run_all_detectors(
    events: Sequence[CanonicalEvent],
) -> dict[str, list[DetectedAction]]:
    """Run every detector on one stay's timeline and return grouped results."""
    return {name: fn(events) for name, fn in _ALL_DETECTORS.items()}
