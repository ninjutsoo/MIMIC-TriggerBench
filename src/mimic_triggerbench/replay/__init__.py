"""Replay environment and structured tools (Phase 7).

The replay environment is the only allowed agent interface to timeline data.
All tools enforce a hard temporal ceiling at ``decision_time``.
"""

from .environment import ReplayEnvironment, ReplayEnvironmentError
from .tools import (
    TOOL_REGISTRY,
    get_active_infusions,
    get_recent_fluids,
    get_recent_labs,
    get_recent_meds,
    get_recent_procedures,
    get_recent_vitals,
    get_trend,
)

__all__ = [
    "ReplayEnvironment",
    "ReplayEnvironmentError",
    "TOOL_REGISTRY",
    "get_active_infusions",
    "get_recent_fluids",
    "get_recent_labs",
    "get_recent_meds",
    "get_recent_procedures",
    "get_recent_vitals",
    "get_trend",
]
