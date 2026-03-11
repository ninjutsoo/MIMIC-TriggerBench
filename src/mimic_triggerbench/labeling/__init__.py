"""
Labeling subpackage: task specifications, trigger definitions, and episode generation.

Phase 2 exposes:
- TaskSpec pydantic model and YAML loader
- load_task_spec / load_all_task_specs helpers
"""

from .task_spec_models import (
    ActionFamily,
    EvidenceType,
    ExclusionRule,
    NegativeWindowDef,
    TaskSpec,
    TriggerDef,
)
from .task_spec_loader import load_task_spec, load_all_task_specs

__all__ = [
    "ActionFamily",
    "EvidenceType",
    "ExclusionRule",
    "NegativeWindowDef",
    "TaskSpec",
    "TriggerDef",
    "load_task_spec",
    "load_all_task_specs",
]
