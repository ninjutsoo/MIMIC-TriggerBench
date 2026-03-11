"""Load and validate versioned YAML task specifications (Phase 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from .task_spec_models import TaskSpec

_SPECS_DIR = Path(__file__).resolve().parent / "task_specs"

_KNOWN_TASKS = ("hyperkalemia", "hypoglycemia", "hypotension")


def _find_spec_file(task_name: str, version: str | None = None) -> Path:
    """Return the path to the YAML file for *task_name* (and optional *version*).

    When *version* is ``None`` the highest available version is returned.
    """
    candidates = sorted(_SPECS_DIR.glob(f"{task_name}_v*.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"No spec files found for task {task_name!r} in {_SPECS_DIR}"
        )
    if version is not None:
        target = _SPECS_DIR / f"{task_name}_{version}.yaml"
        if not target.exists():
            raise FileNotFoundError(
                f"Spec file not found: {target}  (available: {[p.name for p in candidates]})"
            )
        return target
    return candidates[-1]


def load_task_spec(task_name: str, version: str | None = None) -> TaskSpec:
    """Load a single task spec by name, optionally pinning to a version string like ``'v0.1'``."""
    path = _find_spec_file(task_name, version)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TaskSpec.model_validate(raw)


def load_all_task_specs(version: str | None = None) -> Dict[str, TaskSpec]:
    """Load all known task specs and return a ``{task_name: TaskSpec}`` dict."""
    specs: Dict[str, TaskSpec] = {}
    for task in _KNOWN_TASKS:
        specs[task] = load_task_spec(task, version)
    return specs


def list_available_specs() -> List[str]:
    """Return the filenames of all YAML specs in the task_specs directory."""
    return sorted(p.name for p in _SPECS_DIR.glob("*.yaml"))
