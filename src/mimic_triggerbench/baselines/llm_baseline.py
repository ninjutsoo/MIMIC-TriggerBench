"""Single-pass LLM baseline — local Hugging Face causal LM on one GPU (Baseline C, Phase 8).

One generation per episode with a compact structured summary.  No tool use by the model.
Configure cache via ``HF_HOME`` and model via ``TRIGGERBENCH_HF_MODEL_ID`` (see ``hf_local_llm``).

No API or JSON fallbacks: failures propagate for real evaluation runs.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from mimic_triggerbench.labeling.episode_models import Episode
from mimic_triggerbench.labeling.task_spec_models import TaskSpec
from mimic_triggerbench.replay import ReplayEnvironment
from mimic_triggerbench.schemas.outputs import (
    BenchmarkOutput,
    EpisodeInput,
    EvidenceItem,
    ToolTraceItem,
)

from .hf_local_llm import (
    LocalHFGenerator,
    get_default_model_id,
    get_shared_local_generator,
    parse_benchmark_json_response,
)

logger = logging.getLogger(__name__)


def _build_prompt(episode: Episode, spec: TaskSpec, context_summary: str) -> str:
    return (
        f"You are a clinical decision support system. Given the following patient context, "
        f"determine whether the clinical trigger for {spec.display_name} is active and "
        f"recommend appropriate actions.\n\n"
        f"Task: {spec.display_name}\n"
        f"Trigger: {spec.trigger.description}\n"
        f"Decision time: {episode.decision_time.isoformat()}\n"
        f"Context window start: {episode.context_start.isoformat()}\n\n"
        f"Recent clinical data:\n{context_summary}\n\n"
        f"Respond with ONLY a single JSON object (no markdown, no preamble) with exactly these keys:\n"
        f"- trigger_detected (boolean)\n"
        f"- urgency_level (one of: low, medium, high, critical)\n"
        f"- recommended_action_families (array of strings; choose from: {[af.name for af in spec.action_families]})\n"
        f"- recommended_next_steps (array of strings)\n"
        f"- confidence (number 0-1)\n"
        f"- abstain (boolean)\n"
        f"- abstain_reason (string or null)\n"
        f"- reasoning (string)\n"
    )


def _gather_context(env: ReplayEnvironment, episode: Episode, spec: TaskSpec) -> tuple[str, list, list]:
    """Gather a compact context summary via replay tools; return (summary_text, evidence, trace)."""
    ep_input = EpisodeInput(
        episode_id=episode.episode_id,
        task_name=episode.task_name,  # type: ignore[arg-type]
        stay_id=episode.stay_id,
        hadm_id=episode.hadm_id,
        subject_id=episode.subject_id,
        decision_time=episode.decision_time,
        context_start=episode.context_start,
        mandatory_evidence_types=list(episode.mandatory_evidence_types),
        accepted_action_families=list(episode.accepted_action_families),
    )
    env.load_episode(ep_input)

    evidence: List[EvidenceItem] = []
    tool_trace: List[ToolTraceItem] = []
    lines: List[str] = []

    signal = spec.trigger.signal
    if signal in ("potassium", "glucose", "sodium", "creatinine", "bun", "lactate", "bicarbonate", "chloride"):
        tr = env.call_tool("get_recent_labs", {"lab_names": [signal], "hours_back": 24})
    else:
        tr = env.call_tool("get_recent_vitals", {"vital_names": [signal], "minutes_back": 120})

    tool_trace.append(ToolTraceItem(
        tool_name=tr.tool_name,
        arguments={"signal": signal},
        returned_count=tr.result_count,
    ))

    for row in tr.results[-5:]:
        lines.append(f"  {row.canonical_name}: {row.value} {row.unit or ''} at {row.event_time.isoformat()}")
        evidence.append(EvidenceItem(
            source_table=row.source_table,
            canonical_name=row.canonical_name,
            event_time=row.event_time,
            value=row.value if row.value is not None else "NA",
            why_relevant=f"Recent {signal} for LLM baseline context.",
        ))

    tr_vitals = env.call_tool("get_recent_vitals", {"vital_names": ["map", "heart_rate"], "minutes_back": 60})
    tool_trace.append(ToolTraceItem(
        tool_name=tr_vitals.tool_name,
        arguments={"vital_names": ["map", "heart_rate"]},
        returned_count=tr_vitals.result_count,
    ))
    for row in tr_vitals.results[-3:]:
        lines.append(f"  {row.canonical_name}: {row.value} {row.unit or ''} at {row.event_time.isoformat()}")

    return "\n".join(lines) if lines else "(no data)", evidence, tool_trace


class LLMBaseline:
    """Single-pass local HF LLM baseline: one ``generate`` call per episode."""

    def __init__(
        self,
        env: ReplayEnvironment,
        task_specs: Dict[str, TaskSpec],
        *,
        generator: Optional[LocalHFGenerator] = None,
        model_id: Optional[str] = None,
    ) -> None:
        self._env = env
        self._specs = task_specs
        self._generator_override = generator
        self._model_id = model_id or get_default_model_id()
        self.generation_count = 0

    def _generator(self) -> LocalHFGenerator:
        if self._generator_override is not None:
            return self._generator_override
        return get_shared_local_generator(model_id=self._model_id)

    def run_episode(self, episode: Episode) -> BenchmarkOutput:
        spec = self._specs[episode.task_name]
        context_summary, evidence, tool_trace = _gather_context(self._env, episode, spec)
        prompt = _build_prompt(episode, spec, context_summary)

        gen = self._generator()
        raw = gen.generate(prompt, temperature=0.0)
        self.generation_count += 1

        parsed = parse_benchmark_json_response(raw, spec, episode)

        tool_trace.append(ToolTraceItem(
            tool_name="hf_local_generate",
            arguments={"model_id": self._model_id, "baseline": "llm_single_pass"},
            returned_count=1,
        ))

        return BenchmarkOutput(
            episode_id=episode.episode_id,
            task_name=episode.task_name,  # type: ignore[arg-type]
            trigger_detected=parsed["trigger_detected"],
            trigger_type=episode.trigger_type,
            decision_time=episode.decision_time,
            urgency_level=parsed["urgency_level"],  # type: ignore[arg-type]
            recommended_next_steps=parsed["recommended_next_steps"],
            recommended_action_families=parsed["recommended_action_families"],
            evidence=evidence,
            missing_information=[],
            abstain=parsed["abstain"],
            abstain_reason=parsed["abstain_reason"],
            confidence=parsed["confidence"],
            tool_trace=tool_trace,
        )

    def run_all(self, episodes: Sequence[Episode]) -> List[BenchmarkOutput]:
        return [self.run_episode(ep) for ep in episodes]
