"""Action extraction feasibility checkpoint (Phase 3.5).

Provides:
- Deterministic detectors for insulin+dextrose, vasopressor start/escalation,
  fluid bolus, and dialysis start (including infusion rate normalization)
- Coverage reporting with aggregate stats, sampled review sets, and
  go/no-go feasibility decisions
"""

from .detectors import (
    DetectedAction,
    detect_insulin_dextrose_pairing,
    detect_vasopressor_actions,
    detect_fluid_bolus,
    detect_dialysis_start,
    run_all_detectors,
    normalize_vaso_rate,
)
from .coverage_report import (
    ActionFamilyStats,
    FeasibilityThreshold,
    FeasibilityDecision,
    DEFAULT_THRESHOLDS,
    evaluate_feasibility,
    sample_review_set,
    write_coverage_report_md,
    write_coverage_report_json,
    run_feasibility_checkpoint,
    write_feasibility_reports,
)

__all__ = [
    "DetectedAction",
    "detect_insulin_dextrose_pairing",
    "detect_vasopressor_actions",
    "detect_fluid_bolus",
    "detect_dialysis_start",
    "run_all_detectors",
    "normalize_vaso_rate",
    "ActionFamilyStats",
    "FeasibilityThreshold",
    "FeasibilityDecision",
    "DEFAULT_THRESHOLDS",
    "evaluate_feasibility",
    "sample_review_set",
    "write_coverage_report_md",
    "write_coverage_report_json",
    "run_feasibility_checkpoint",
    "write_feasibility_reports",
]
