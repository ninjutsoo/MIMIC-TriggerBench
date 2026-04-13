"""Baseline runners for the benchmark (Phase 8).

All baselines share the same ``EpisodeInput`` -> ``BenchmarkOutput`` contract,
use the same frozen schema, and can be scored identically via the evaluation harness.
"""

from .rule_baseline import RuleBaseline
from .tabular_baseline import TabularBaseline
from .llm_baseline import LLMBaseline
from .rag_baseline import RAGBaseline
from .feature_builder import FeatureBuilder, FeatureArtifact
from .hf_local_llm import (
    LocalHFGenerator,
    LocalHFGeneratorError,
    get_default_model_id,
    get_shared_local_generator,
    parse_benchmark_json_response,
    reset_shared_local_generator_for_tests,
)

__all__ = [
    "RuleBaseline",
    "TabularBaseline",
    "LLMBaseline",
    "RAGBaseline",
    "FeatureBuilder",
    "FeatureArtifact",
    "LocalHFGenerator",
    "LocalHFGeneratorError",
    "get_default_model_id",
    "get_shared_local_generator",
    "parse_benchmark_json_response",
    "reset_shared_local_generator_for_tests",
]
