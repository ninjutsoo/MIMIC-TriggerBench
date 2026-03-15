"""Coverage reports, feasibility gate, and checkpoint orchestrator (Phase 3.5).

Aggregates detector outputs across stays, produces human-readable and
machine-readable summaries, samples review sets, evaluates go/no-go
thresholds, and orchestrates the full checkpoint pipeline.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from mimic_triggerbench.timeline.models import CanonicalEvent
from .detectors import DetectedAction, run_all_detectors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Aggregate statistics per action family
# ---------------------------------------------------------------------------

@dataclass
class ActionFamilyStats:
    """Aggregate statistics for one action family across all processed stays."""

    action_family: str
    detected_events: int = 0
    unique_stays: int = 0
    unique_patients: int = 0
    failure_patterns: dict[str, int] = field(default_factory=dict)

    _stay_ids: set[int] = field(default_factory=set, repr=False)
    _subject_ids: set[int] = field(default_factory=set, repr=False)

    def add_detections(self, actions: Sequence[DetectedAction]) -> None:
        for a in actions:
            self.detected_events += 1
            self._stay_ids.add(a.stay_id)
            self._subject_ids.add(a.subject_id)
        self.unique_stays = len(self._stay_ids)
        self.unique_patients = len(self._subject_ids)

    def record_failure(self, pattern: str, count: int = 1) -> None:
        self.failure_patterns[pattern] = self.failure_patterns.get(pattern, 0) + count

    def to_dict(self) -> dict:
        return {
            "action_family": self.action_family,
            "detected_events": self.detected_events,
            "unique_stays": self.unique_stays,
            "unique_patients": self.unique_patients,
            "failure_patterns": self.failure_patterns,
        }


# ---------------------------------------------------------------------------
# Review-set sampling
# ---------------------------------------------------------------------------

def sample_review_set(
    detections: list[DetectedAction],
    n: int = 25,
    *,
    seed: int = 42,
) -> list[DetectedAction]:
    """Sample up to *n* detections for manual review (deterministic)."""
    if len(detections) <= n:
        return list(detections)
    rng = random.Random(seed)
    return rng.sample(detections, n)


# ---------------------------------------------------------------------------
# Go / no-go feasibility thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeasibilityThreshold:
    """Pre-frozen acceptance gate for one action family."""

    action_family: str
    min_detected_events: int = 10
    min_unique_stays: int = 5
    min_unique_patients: int = 3

    def evaluate(self, stats: ActionFamilyStats) -> bool:
        return (
            stats.detected_events >= self.min_detected_events
            and stats.unique_stays >= self.min_unique_stays
            and stats.unique_patients >= self.min_unique_patients
        )


DEFAULT_THRESHOLDS: dict[str, FeasibilityThreshold] = {
    "insulin_dextrose": FeasibilityThreshold("insulin_dextrose", min_detected_events=10, min_unique_stays=5, min_unique_patients=3),
    "vasopressor_start": FeasibilityThreshold("vasopressor_start", min_detected_events=20, min_unique_stays=10, min_unique_patients=5),
    "vasopressor_escalation": FeasibilityThreshold("vasopressor_escalation", min_detected_events=10, min_unique_stays=5, min_unique_patients=3),
    "fluid_bolus": FeasibilityThreshold("fluid_bolus", min_detected_events=20, min_unique_stays=10, min_unique_patients=5),
    "dialysis_start": FeasibilityThreshold("dialysis_start", min_detected_events=5, min_unique_stays=3, min_unique_patients=2),
}


# ---------------------------------------------------------------------------
# Feasibility decision record
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityDecision:
    """Go / no-go decision for one action family."""

    action_family: str
    threshold: FeasibilityThreshold
    stats: ActionFamilyStats
    passed: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "action_family": self.action_family,
            "passed": self.passed,
            "note": self.note,
            "threshold": {
                "min_detected_events": self.threshold.min_detected_events,
                "min_unique_stays": self.threshold.min_unique_stays,
                "min_unique_patients": self.threshold.min_unique_patients,
            },
            "stats": self.stats.to_dict(),
        }


def evaluate_feasibility(
    all_stats: dict[str, ActionFamilyStats],
    thresholds: Optional[dict[str, FeasibilityThreshold]] = None,
) -> list[FeasibilityDecision]:
    """Evaluate go/no-go for every action family."""
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    decisions: list[FeasibilityDecision] = []
    for family, stats in sorted(all_stats.items()):
        threshold = thresholds.get(family)
        if threshold is None:
            threshold = FeasibilityThreshold(family)
        passed = threshold.evaluate(stats)
        note = "PASS" if passed else "BELOW THRESHOLD — consider down-scoping"
        decisions.append(FeasibilityDecision(
            action_family=family,
            threshold=threshold,
            stats=stats,
            passed=passed,
            note=note,
        ))
    return decisions


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def write_coverage_report_md(
    decisions: list[FeasibilityDecision],
    review_sets: dict[str, list[DetectedAction]],
    out_path: Path,
) -> None:
    """Write a Markdown feasibility coverage report."""
    lines: list[str] = [
        "# Action Extraction Feasibility Report (Phase 3.5)",
        "",
        "## Summary",
        "",
        "| Action Family | Events | Stays | Patients | Gate |",
        "|---------------|-------:|------:|---------:|------|",
    ]
    for d in decisions:
        gate = "PASS" if d.passed else "FAIL"
        lines.append(
            f"| {d.action_family} | {d.stats.detected_events} "
            f"| {d.stats.unique_stays} | {d.stats.unique_patients} "
            f"| **{gate}** |"
        )
    lines.append("")

    for d in decisions:
        lines.append(f"## {d.action_family}")
        lines.append("")
        lines.append(f"- Detected events: {d.stats.detected_events}")
        lines.append(f"- Unique stays: {d.stats.unique_stays}")
        lines.append(f"- Unique patients: {d.stats.unique_patients}")
        lines.append(f"- Gate: **{'PASS' if d.passed else 'FAIL'}** — {d.note}")
        lines.append("")

        if d.stats.failure_patterns:
            lines.append("### Failure patterns")
            lines.append("")
            for pat, cnt in d.stats.failure_patterns.items():
                lines.append(f"- {pat}: {cnt}")
            lines.append("")

        review = review_sets.get(d.action_family, [])
        if review:
            lines.append(f"### Sampled detections for review ({len(review)})")
            lines.append("")
            for r in review[:10]:
                lines.append(f"- stay={r.stay_id} t={r.detection_time.isoformat()} "
                             f"details={json.dumps(r.details, default=str)}")
            if len(review) > 10:
                lines.append(f"- ... and {len(review) - 10} more")
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_coverage_report_json(
    decisions: list[FeasibilityDecision],
    review_sets: dict[str, list[DetectedAction]],
    out_path: Path,
) -> None:
    """Write a machine-readable JSON feasibility report."""
    payload = {
        "decisions": [d.to_dict() for d in decisions],
        "review_sets": {
            family: [a.to_dict() for a in actions]
            for family, actions in review_sets.items()
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Checkpoint orchestrator (merged from checkpoint.py)
# ---------------------------------------------------------------------------

def run_feasibility_checkpoint(
    timelines: dict[int, list[CanonicalEvent]],
    *,
    thresholds: Optional[dict[str, FeasibilityThreshold]] = None,
    review_sample_n: int = 25,
    review_seed: int = 42,
) -> tuple[list[FeasibilityDecision], dict[str, list[DetectedAction]]]:
    """Run the full feasibility checkpoint across all provided timelines.

    Returns ``(decisions, review_sets)`` where *review_sets* maps
    action-family name → sampled detections.
    """
    all_detections: dict[str, list[DetectedAction]] = {}
    stats: dict[str, ActionFamilyStats] = {}

    for stay_id, events in timelines.items():
        stay_results = run_all_detectors(events)
        for detector_name, actions in stay_results.items():
            for action in actions:
                family = action.action_family
                if family not in all_detections:
                    all_detections[family] = []
                all_detections[family].append(action)

    for family, actions in all_detections.items():
        s = ActionFamilyStats(action_family=family)
        s.add_detections(actions)
        stats[family] = s

    for expected in ("insulin_dextrose", "vasopressor_start",
                     "vasopressor_escalation", "fluid_bolus", "dialysis_start"):
        if expected not in stats:
            stats[expected] = ActionFamilyStats(action_family=expected)

    decisions = evaluate_feasibility(stats, thresholds)

    review_sets: dict[str, list[DetectedAction]] = {}
    for family, actions in all_detections.items():
        review_sets[family] = sample_review_set(
            actions, n=review_sample_n, seed=review_seed,
        )

    logger.info(
        "Feasibility checkpoint: %d families evaluated, %d passed",
        len(decisions),
        sum(1 for d in decisions if d.passed),
    )
    return decisions, review_sets


def write_feasibility_reports(
    decisions: list[FeasibilityDecision],
    review_sets: dict[str, list[DetectedAction]],
    output_dir: Path,
) -> None:
    """Write both Markdown and JSON feasibility reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "feasibility_report.md"
    json_path = output_dir / "feasibility_report.json"
    write_coverage_report_md(decisions, review_sets, md_path)
    write_coverage_report_json(decisions, review_sets, json_path)
    logger.info("Feasibility reports written to %s", output_dir)
