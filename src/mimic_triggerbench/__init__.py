"""
Top-level package for the MIMIC-TriggerBench benchmark.

This module intentionally keeps runtime side effects minimal; most functionality
is organized into subpackages:

- config
- data_access
- labeling
- replay
- baselines
- agent
- evaluation
"""

__all__ = [
    "config",
    "data_access",
    "labeling",
    "replay",
    "baselines",
    "agent",
    "evaluation",
]

