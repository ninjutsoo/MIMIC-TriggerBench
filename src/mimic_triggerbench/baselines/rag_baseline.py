"""Simple RAG baseline — fixed retrieval pack + one-shot local HF model (Baseline D, Phase 8).

Retrieves a broader context pack from the timeline and protocol text,
then issues a single local ``generate`` call.  Same JSON contract as the LLM baseline.

No API or JSON fallbacks.
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


def _build_protocol_text(spec: TaskSpec) -> str:
    """Build a structured protocol reference from the task spec."""
    lines = [
        f"PROTOCOL: {spec.display_name}",
        f"Trigger: {spec.trigger.description}",
        f"Signal: {spec.trigger.signal} {spec.trigger.operator.value} {spec.trigger.threshold} {spec.trigger.unit}",
        "",
        "Accepted action families:",
    ]
    for af in spec.action_families:
        primary = " [PRIMARY]" if af.is_primary else ""
        lines.append(f"  - {af.name}{primary}: {af.description}")
        if af.concrete_examples:
            lines.append(f"    Examples: {', '.join(af.concrete_examples)}")
    lines.append("")
    lines.append("Required evidence types:")
    for et in spec.evidence_types:
        req = "required" if et.required else "optional"
        lines.append(f"  - {et.name} (lookback {et.lookback_hours}h, {req})")
    return "\n".join(lines)


def _gather_rag_context(
    env: ReplayEnvironment,
    episode: Episode,
    spec: TaskSpec,
) -> tuple[str, List[EvidenceItem], List[ToolTraceItem]]:
    """Broader context retrieval than the plain LLM baseline."""
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
    sections: List[str] = []

    signal = spec.trigger.signal
    if signal in ("potassium", "glucose", "sodium", "creatinine", "bun", "lactate", "bicarbonate", "chloride"):
        tr = env.call_tool("get_recent_labs", {"lab_names": [signal], "hours_back": 24})
    else:
        tr = env.call_tool("get_recent_vitals", {"vital_names": [signal], "minutes_back": 120})
    tool_trace.append(ToolTraceItem(tool_name=tr.tool_name, arguments={"signal": signal}, returned_count=tr.result_count))
    section_lines = [f"[{spec.trigger.signal} values]"]
    for row in tr.results:
        section_lines.append(f"  {row.value} {row.unit or ''} at {row.event_time.isoformat()}")
        evidence.append(EvidenceItem(
            source_table=row.source_table, canonical_name=row.canonical_name,
            event_time=row.event_time, value=row.value if row.value is not None else "NA",
            why_relevant=f"RAG context: {signal} history",
        ))
    sections.append("\n".join(section_lines))

    tr_vitals = env.call_tool("get_recent_vitals", {"vital_names": ["map", "heart_rate", "sbp"], "minutes_back": 120})
    tool_trace.append(ToolTraceItem(tool_name=tr_vitals.tool_name, arguments={"vital_names": ["map", "heart_rate", "sbp"]}, returned_count=tr_vitals.result_count))
    if tr_vitals.results:
        vlines = ["[Vitals]"]
        for row in tr_vitals.results[-5:]:
            vlines.append(f"  {row.canonical_name}: {row.value} {row.unit or ''} at {row.event_time.isoformat()}")
        sections.append("\n".join(vlines))

    tr_meds = env.call_tool("get_recent_meds", {"med_classes": [], "hours_back": 12})
    tool_trace.append(ToolTraceItem(tool_name=tr_meds.tool_name, arguments={"hours_back": 12}, returned_count=tr_meds.result_count))
    if tr_meds.results:
        mlines = ["[Recent medications]"]
        for row in tr_meds.results[-5:]:
            mlines.append(f"  {row.canonical_name} at {row.event_time.isoformat()}")
        sections.append("\n".join(mlines))

    tr_inf = env.call_tool("get_active_infusions", {})
    tool_trace.append(ToolTraceItem(tool_name=tr_inf.tool_name, arguments={}, returned_count=tr_inf.result_count))
    if tr_inf.results:
        ilines = ["[Active infusions]"]
        for row in tr_inf.results:
            ilines.append(f"  {row.canonical_name} since {row.event_time.isoformat()}")
        sections.append("\n".join(ilines))

    return "\n\n".join(sections) if sections else "(no data)", evidence, tool_trace


def _build_rag_prompt(episode: Episode, spec: TaskSpec, protocol: str, context: str) -> str:
    return (
        f"You are a clinical decision support system with access to protocol guidelines "
        f"and recent patient data.\n\n"
        f"=== PROTOCOL ===\n{protocol}\n\n"
        f"=== PATIENT DATA (decision_time={episode.decision_time.isoformat()}) ===\n{context}\n\n"
        f"Output ONLY a single JSON object (no markdown, no preamble) with exactly these keys:\n"
        f"- trigger_detected (boolean)\n"
        f"- urgency_level (one of: low, medium, high, critical)\n"
        f"- recommended_action_families (array of strings from: {[af.name for af in spec.action_families]})\n"
        f"- recommended_next_steps (array of strings)\n"
        f"- confidence (number 0-1)\n"
        f"- abstain (boolean)\n"
        f"- abstain_reason (string or null)\n"
    )


class RAGBaseline:
    """Fixed retrieval + protocol text + one local HF generation."""

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
        protocol = _build_protocol_text(spec)
        context, evidence, tool_trace = _gather_rag_context(self._env, episode, spec)
        prompt = _build_rag_prompt(episode, spec, protocol, context)

        gen = self._generator()
        raw = gen.generate(prompt, temperature=0.0)
        self.generation_count += 1

        parsed = parse_benchmark_json_response(raw, spec, episode)

        tool_trace.append(ToolTraceItem(
            tool_name="hf_local_generate",
            arguments={"model_id": self._model_id, "baseline": "rag_one_shot"},
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
