"""Tests for Phase 8 — Baselines and evaluation harness.

Covers:
- Rule baseline: runs on synthetic episode, emits valid BenchmarkOutput
- Tabular baseline: fit/transform split discipline, feature artifact metadata
- Strict JSON parsing for local HF outputs
- LLM / RAG baselines: real local HF forward pass when ``TRIGGERBENCH_RUN_HF_TESTS=1`` and CUDA is available
- Evaluation harness: scoring produces correct metrics
- All outputs validate against the frozen schema
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pytest

from mimic_triggerbench.baselines.feature_builder import (
    FeatureBuilder,
    FeatureArtifact,
    FEATURE_SPEC_VERSION,
    _config_hash,
)
from mimic_triggerbench.baselines.rule_baseline import RuleBaseline
from mimic_triggerbench.baselines.tabular_baseline import TabularBaseline
from mimic_triggerbench.evaluation.scoring import (
    EpisodeScore,
    RunScores,
    aggregate_scores,
    score_episode,
    score_run,
)
from mimic_triggerbench.labeling.episode_models import Episode
from mimic_triggerbench.labeling.task_spec_loader import load_all_task_specs
from mimic_triggerbench.replay import ReplayEnvironment
from mimic_triggerbench.schemas import (
    BenchmarkOutput,
    EpisodeInput,
    validate_benchmark_output,
)
from mimic_triggerbench.timeline.models import CanonicalEvent, generate_event_uid


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _hf_smoke_enabled() -> bool:
    return os.environ.get("TRIGGERBENCH_RUN_HF_TESTS") == "1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2150, 3, 1, 8, 0, 0)
DECISION_TIME = T0 + timedelta(hours=2)
STAY_ID = 100
SUBJECT_ID = 10
HADM_ID = 20


def _ev(
    *,
    offset_min: int = 0,
    source_table: str = "labevents",
    category: str = "lab",
    name: str = "potassium",
    value: float | None = 6.5,
    unit: str | None = "mmol/L",
) -> CanonicalEvent:
    t = T0 + timedelta(minutes=offset_min)
    uid = generate_event_uid(source_table, STAY_ID, t, name, extra_discriminator=str(offset_min))
    return CanonicalEvent(
        subject_id=SUBJECT_ID,
        hadm_id=HADM_ID,
        stay_id=STAY_ID,
        event_time=t,
        event_uid=uid,
        source_table=source_table,
        event_category=category,
        canonical_name=name,
        value_num=value,
        unit=unit,
    )


def _make_timeline() -> Dict[int, List[CanonicalEvent]]:
    events = [
        _ev(offset_min=0, name="potassium", value=5.0, category="lab", source_table="labevents"),
        _ev(offset_min=30, name="potassium", value=6.5, category="lab", source_table="labevents"),
        _ev(offset_min=60, name="potassium", value=6.8, category="lab", source_table="labevents"),
        _ev(offset_min=10, name="map", value=75.0, category="vital", source_table="chartevents", unit="mmHg"),
        _ev(offset_min=50, name="map", value=70.0, category="vital", source_table="chartevents", unit="mmHg"),
        _ev(offset_min=90, name="heart_rate", value=90.0, category="vital", source_table="chartevents", unit="bpm"),
    ]
    return {STAY_ID: events}


def _make_episode(
    trigger_label: bool = True,
    trigger_type: str = "positive",
    task_name: str = "hyperkalemia",
) -> Episode:
    action_families = ["insulin_dextrose", "calcium_gluconate"] if trigger_label else []
    return Episode(
        episode_id=f"test-ep-{task_name}-{trigger_type}",
        task_name=task_name,
        subject_id=SUBJECT_ID,
        hadm_id=HADM_ID,
        stay_id=STAY_ID,
        decision_time=DECISION_TIME,
        context_start=T0,
        trigger_label=trigger_label,
        trigger_type=trigger_type,
        trigger_value=6.8 if trigger_label else None,
        accepted_action_families=action_families,
        observed_action_families=[],
        mandatory_evidence_types=["potassium"],
    )


# ---------------------------------------------------------------------------
# Rule baseline tests
# ---------------------------------------------------------------------------


class TestRuleBaseline:
    def test_positive_episode_valid_output(self):
        timelines = _make_timeline()
        env = ReplayEnvironment(timelines)
        specs = load_all_task_specs()
        baseline = RuleBaseline(env, specs)

        ep = _make_episode(trigger_label=True, trigger_type="positive")
        output = baseline.run_episode(ep)

        assert isinstance(output, BenchmarkOutput)
        validate_benchmark_output(output.model_dump(mode="json"))
        assert output.trigger_detected is True
        assert output.abstain is False
        assert len(output.recommended_action_families) > 0
        assert len(output.evidence) > 0
        assert len(output.tool_trace) > 0

    def test_negative_episode_abstains(self):
        timelines = _make_timeline()
        env = ReplayEnvironment(timelines)
        specs = load_all_task_specs()
        baseline = RuleBaseline(env, specs)

        ep = _make_episode(trigger_label=False, trigger_type="negative")
        output = baseline.run_episode(ep)

        validate_benchmark_output(output.model_dump(mode="json"))
        assert output.trigger_detected is False
        assert output.abstain is True
        assert output.recommended_action_families == []


# ---------------------------------------------------------------------------
# Feature builder tests
# ---------------------------------------------------------------------------


class TestFeatureBuilder:
    def test_fit_transform_returns_correct_shapes(self):
        timelines = _make_timeline()
        fb = FeatureBuilder(timelines, split_seed=42)

        episodes = [
            _make_episode(trigger_label=True),
            _make_episode(trigger_label=False, trigger_type="negative"),
        ]
        X, y = fb.fit_transform(episodes)

        assert X.shape[0] == 2
        assert X.shape[1] > 0
        assert y.shape == (2,)
        assert set(y) == {0, 1}

    def test_transform_uses_fitted_preprocessing(self):
        timelines = _make_timeline()
        fb = FeatureBuilder(timelines, split_seed=42)

        train = [_make_episode(trigger_label=True), _make_episode(trigger_label=False, trigger_type="negative")]
        fb.fit_transform(train)

        val = [_make_episode(trigger_label=True)]
        X_val, _ = fb.transform(val)
        assert X_val.shape[0] == 1
        assert X_val.shape[1] == fb.artifact.n_features

    def test_transform_before_fit_raises(self):
        fb = FeatureBuilder({})
        with pytest.raises(RuntimeError, match="fit_transform"):
            fb.transform([_make_episode()])

    def test_artifact_metadata(self):
        timelines = _make_timeline()
        fb = FeatureBuilder(timelines, split_seed=42)
        fb.fit_transform([_make_episode()])

        art = fb.artifact
        assert art.feature_spec_version == FEATURE_SPEC_VERSION
        assert art.config_hash == _config_hash()
        assert art.split_seed == 42
        assert art.n_features > 0
        assert len(art.feature_names) == art.n_features


# ---------------------------------------------------------------------------
# Tabular baseline tests
# ---------------------------------------------------------------------------


class TestTabularBaseline:
    def test_fit_and_predict(self):
        timelines = _make_timeline()
        fb = FeatureBuilder(timelines, split_seed=42)
        tb = TabularBaseline(fb, model_type="xgboost", seed=42)

        train = [
            _make_episode(trigger_label=True),
            _make_episode(trigger_label=False, trigger_type="negative"),
            _make_episode(trigger_label=True),
            _make_episode(trigger_label=False, trigger_type="negative"),
        ]
        tb.fit(train)
        outputs = tb.predict(train[:2])

        assert len(outputs) == 2
        for out in outputs:
            validate_benchmark_output(out.model_dump(mode="json"))

    def test_predict_before_fit_raises(self):
        fb = FeatureBuilder({})
        tb = TabularBaseline(fb)
        with pytest.raises(RuntimeError):
            tb.predict([_make_episode()])

    def test_logistic_regression_variant(self):
        timelines = _make_timeline()
        fb = FeatureBuilder(timelines, split_seed=42)
        tb = TabularBaseline(fb, model_type="logistic", seed=42)

        train = [
            _make_episode(trigger_label=True),
            _make_episode(trigger_label=False, trigger_type="negative"),
            _make_episode(trigger_label=True),
            _make_episode(trigger_label=False, trigger_type="negative"),
        ]
        tb.fit(train)
        outputs = tb.predict(train[:1])
        assert len(outputs) == 1
        validate_benchmark_output(outputs[0].model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Evaluation harness tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_score_positive_episode_perfect(self):
        ep = _make_episode(trigger_label=True)
        output = BenchmarkOutput(
            episode_id=ep.episode_id,
            task_name="hyperkalemia",
            trigger_detected=True,
            trigger_type="positive",
            decision_time=ep.decision_time,
            urgency_level="high",
            recommended_next_steps=["treat"],
            recommended_action_families=["insulin_dextrose", "calcium_gluconate"],
            evidence=[],
            missing_information=[],
            abstain=False,
            abstain_reason=None,
            confidence=0.9,
            tool_trace=[],
        )
        score = score_episode(output, ep)
        assert score.trigger_correct is True
        assert score.abstain_correct is True
        assert score.action_family_f1 == 1.0

    def test_score_negative_episode_correct_abstain(self):
        ep = _make_episode(trigger_label=False, trigger_type="negative")
        output = BenchmarkOutput(
            episode_id=ep.episode_id,
            task_name="hyperkalemia",
            trigger_detected=False,
            trigger_type="negative",
            decision_time=ep.decision_time,
            urgency_level="low",
            recommended_next_steps=[],
            recommended_action_families=[],
            evidence=[],
            missing_information=[],
            abstain=True,
            abstain_reason="negative",
            confidence=0.5,
            tool_trace=[],
        )
        score = score_episode(output, ep)
        assert score.trigger_correct is True
        assert score.abstain_correct is True
        assert score.action_family_f1 == 1.0

    def test_score_run_and_aggregate(self):
        episodes = [
            _make_episode(trigger_label=True),
            _make_episode(trigger_label=False, trigger_type="negative"),
        ]
        outputs = []
        for ep in episodes:
            is_pos = ep.trigger_label
            outputs.append(BenchmarkOutput(
                episode_id=ep.episode_id,
                task_name="hyperkalemia",
                trigger_detected=is_pos,
                trigger_type=ep.trigger_type,
                decision_time=ep.decision_time,
                urgency_level="high" if is_pos else "low",
                recommended_next_steps=[],
                recommended_action_families=list(ep.accepted_action_families),
                evidence=[],
                missing_information=[],
                abstain=not is_pos,
                abstain_reason=None if is_pos else "neg",
                confidence=0.9 if is_pos else 0.5,
                tool_trace=[],
            ))

        run = score_run("test_system", outputs, episodes)
        assert run.n == 2
        assert run.trigger_accuracy == 1.0
        assert run.abstain_accuracy == 1.0

        by_task = aggregate_scores(run, by="task_name")
        assert "hyperkalemia" in by_task
        assert by_task["hyperkalemia"]["trigger_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Local HF JSON parsing (no GPU)
# ---------------------------------------------------------------------------


class TestParseBenchmarkJson:
    def test_valid_json(self) -> None:
        from mimic_triggerbench.baselines.hf_local_llm import parse_benchmark_json_response

        spec = load_all_task_specs()["hyperkalemia"]
        ep = _make_episode(trigger_label=True)
        text = (
            '{"trigger_detected": true, "urgency_level": "high", '
            '"recommended_action_families": ["insulin_dextrose"], '
            '"recommended_next_steps": ["recheck K"], "confidence": 0.9, '
            '"abstain": false, "abstain_reason": null}'
        )
        out = parse_benchmark_json_response(text, spec, ep)
        assert out["trigger_detected"] is True
        assert out["urgency_level"] == "high"
        assert "insulin_dextrose" in out["recommended_action_families"]

    def test_invalid_json_raises(self) -> None:
        from mimic_triggerbench.baselines.hf_local_llm import (
            LocalHFGeneratorError,
            parse_benchmark_json_response,
        )

        spec = load_all_task_specs()["hyperkalemia"]
        ep = _make_episode()
        with pytest.raises(LocalHFGeneratorError):
            parse_benchmark_json_response("not json {", spec, ep)


# ---------------------------------------------------------------------------
# LLM / RAG baselines — real HF model (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _hf_smoke_enabled(),
    reason="Set TRIGGERBENCH_RUN_HF_TESTS=1 to run local HF baseline smoke tests",
)
@pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA required for local HF model",
)
class TestLLMBaseline:
    def test_single_episode(self) -> None:
        from mimic_triggerbench.baselines.hf_local_llm import reset_shared_local_generator_for_tests
        from mimic_triggerbench.baselines.llm_baseline import LLMBaseline

        reset_shared_local_generator_for_tests()
        try:
            timelines = _make_timeline()
            env = ReplayEnvironment(timelines)
            specs = load_all_task_specs()
            baseline = LLMBaseline(env, specs)

            ep = _make_episode(trigger_label=True)
            output = baseline.run_episode(ep)
            validate_benchmark_output(output.model_dump(mode="json"))
            assert baseline.generation_count == 1
        finally:
            reset_shared_local_generator_for_tests()


@pytest.mark.skipif(
    not _hf_smoke_enabled(),
    reason="Set TRIGGERBENCH_RUN_HF_TESTS=1 to run local HF baseline smoke tests",
)
@pytest.mark.skipif(
    not _cuda_available(),
    reason="CUDA required for local HF model",
)
class TestRAGBaseline:
    def test_single_episode(self) -> None:
        from mimic_triggerbench.baselines.hf_local_llm import reset_shared_local_generator_for_tests
        from mimic_triggerbench.baselines.rag_baseline import RAGBaseline

        reset_shared_local_generator_for_tests()
        try:
            timelines = _make_timeline()
            env = ReplayEnvironment(timelines)
            specs = load_all_task_specs()
            baseline = RAGBaseline(env, specs)

            ep = _make_episode(trigger_label=True)
            output = baseline.run_episode(ep)
            validate_benchmark_output(output.model_dump(mode="json"))
            assert baseline.generation_count == 1
        finally:
            reset_shared_local_generator_for_tests()
