"""Replay environment: the sole interface between agents/runners and timeline data (Phase 7).

The environment holds per-stay canonical timelines and exposes a set of
structured, JSON-returning tools.  Every tool call is timestamp-bounded:
the ``decision_time`` of the loaded episode acts as a hard ceiling that
no tool can exceed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from mimic_triggerbench.schemas.outputs import EpisodeInput, ToolResult
from mimic_triggerbench.timeline.models import CanonicalEvent

from .tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)


class ReplayEnvironmentError(RuntimeError):
    """Raised on misuse of the replay environment (no episode loaded, bad tool name, etc.)."""


class ReplayEnvironment:
    """Timestamp-bounded replay environment for benchmark episodes.

    Usage::

        env = ReplayEnvironment(timelines)
        env.load_episode(episode_input)
        result = env.call_tool("get_recent_labs", {
            "lab_names": ["potassium"],
            "hours_back": 24,
        })
    """

    def __init__(self, timelines: Dict[int, List[CanonicalEvent]]) -> None:
        self._timelines = timelines
        self._episode: Optional[EpisodeInput] = None
        self._stay_events: List[CanonicalEvent] = []

    @property
    def episode(self) -> Optional[EpisodeInput]:
        return self._episode

    def load_episode(self, episode: EpisodeInput) -> None:
        """Set the active episode.  All subsequent tool calls are scoped to
        this episode's ``stay_id`` and ``decision_time``."""
        self._episode = episode
        self._stay_events = list(self._timelines.get(episode.stay_id, []))
        logger.info(
            "Loaded episode %s (stay=%d, decision_time=%s, %d events)",
            episode.episode_id,
            episode.stay_id,
            episode.decision_time.isoformat(),
            len(self._stay_events),
        )

    def available_tools(self) -> List[str]:
        """Return the names of all registered tools."""
        return sorted(TOOL_REGISTRY.keys())

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Dispatch a tool call and return a validated :class:`ToolResult`.

        ``arguments`` should contain the tool's keyword arguments **except**
        ``decision_time`` (injected automatically) and the raw event list
        (provided by the environment).
        """
        if self._episode is None:
            raise ReplayEnvironmentError("No episode loaded -- call load_episode first.")
        if tool_name not in TOOL_REGISTRY:
            raise ReplayEnvironmentError(
                f"Unknown tool {tool_name!r}. Available: {self.available_tools()}"
            )

        tool_fn = TOOL_REGISTRY[tool_name]
        decision_time: datetime = self._episode.decision_time

        call_args: Dict[str, Any] = dict(arguments)
        call_args["decision_time"] = decision_time
        call_args.setdefault("before_time", decision_time)

        result: ToolResult = tool_fn(self._stay_events, **call_args)  # type: ignore[operator]
        return result
