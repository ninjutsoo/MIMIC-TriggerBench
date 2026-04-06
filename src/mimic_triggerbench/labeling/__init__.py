"""
Labeling subpackage: task specifications, trigger definitions, episode generation,
and deterministic label assignment.

Phase 2 exposes: TaskSpec pydantic model and YAML loader.
Phase 5 exposes: Episode model, trigger evaluators, and episode generators.
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
from .episode_models import Episode, generate_episode_id, NAMESPACE_EPISODE
from .triggers import evaluate_trigger, TriggerResult
from .episode_generation import (
    generate_episodes_for_stay,
    generate_all_episodes,
    episodes_to_records,
)

__all__ = [
    "ActionFamily",
    "EvidenceType",
    "ExclusionRule",
    "NegativeWindowDef",
    "TaskSpec",
    "TriggerDef",
    "load_task_spec",
    "load_all_task_specs",
    "Episode",
    "generate_episode_id",
    "NAMESPACE_EPISODE",
    "evaluate_trigger",
    "TriggerResult",
    "generate_episodes_for_stay",
    "generate_all_episodes",
    "episodes_to_records",
]
