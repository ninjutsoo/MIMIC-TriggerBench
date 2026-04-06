from __future__ import annotations

from typing import Any, Dict

from pydantic import ValidationError

from .outputs import BenchmarkOutput, EpisodeInput, ToolResult


class OutputSchemaError(ValueError):
    """Raised when a runner emits malformed benchmark output."""


def validate_benchmark_output(payload: Dict[str, Any]) -> BenchmarkOutput:
    """Validate and parse the Phase 6.5 frozen benchmark output payload.

    Runners should call this immediately before emitting JSON.
    """
    try:
        return BenchmarkOutput.model_validate(payload)
    except ValidationError as e:
        raise OutputSchemaError(str(e)) from e


def validate_episode_input(payload: Dict[str, Any]) -> EpisodeInput:
    """Validate and parse an episode-input payload fed to a runner."""
    try:
        return EpisodeInput.model_validate(payload)
    except ValidationError as e:
        raise OutputSchemaError(str(e)) from e


def validate_tool_result(payload: Dict[str, Any]) -> ToolResult:
    """Validate and parse a replay-tool result envelope."""
    try:
        return ToolResult.model_validate(payload)
    except ValidationError as e:
        raise OutputSchemaError(str(e)) from e

