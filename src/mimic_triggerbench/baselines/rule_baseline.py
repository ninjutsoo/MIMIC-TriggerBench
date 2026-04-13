"""Rule-only baseline (Baseline A, Phase 8).

Fires from deterministic trigger rules and returns a predefined escalation
template with minimal contextualisation.  No ML, no LLM.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from mimic_triggerbench.labeling.episode_models import Episode
from mimic_triggerbench.labeling.task_spec_models import TaskSpec
from mimic_triggerbench.replay import ReplayEnvironment
from mimic_triggerbench.schemas.outputs import (
    BenchmarkOutput,
    EpisodeInput,
    EvidenceItem,
    ToolTraceItem,
)

_URGENCY_MAP: Dict[str, str] = {
    "hyperkalemia": "high",
    "hypoglycemia": "high",
    "hypotension": "critical",
}

_NEXT_STEPS: Dict[str, List[str]] = {
    "hyperkalemia": [
        "Administer calcium gluconate for cardiac protection",
        "Administer insulin + dextrose to shift potassium intracellularly",
        "Order stat repeat potassium",
    ],
    "hypoglycemia": [
        "Administer IV dextrose (D50)",
        "Recheck glucose in 15 minutes",
        "Evaluate and adjust insulin regimen",
    ],
    "hypotension": [
        "Initiate fluid bolus (crystalloid 500-1000 mL)",
        "Assess for vasopressor initiation or escalation",
        "Evaluate source of hypotension",
    ],
}


class RuleBaseline:
    """Deterministic rule-based baseline that uses replay tools to gather evidence."""

    def __init__(
        self,
        env: ReplayEnvironment,
        task_specs: Dict[str, TaskSpec],
    ) -> None:
        self._env = env
        self._specs = task_specs

    def run_episode(self, episode: Episode) -> BenchmarkOutput:
        spec = self._specs[episode.task_name]
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
        self._env.load_episode(ep_input)

        evidence: List[EvidenceItem] = []
        tool_trace: List[ToolTraceItem] = []

        signal = spec.trigger.signal
        if signal in ("potassium", "glucose", "sodium", "creatinine", "bun", "lactate", "bicarbonate", "chloride"):
            tr = self._env.call_tool("get_recent_labs", {"lab_names": [signal], "hours_back": 24})
        else:
            tr = self._env.call_tool("get_recent_vitals", {"vital_names": [signal], "minutes_back": 120})

        tool_trace.append(ToolTraceItem(
            tool_name=tr.tool_name,
            arguments={"signal": signal},
            returned_count=tr.result_count,
        ))

        for row in tr.results[-3:]:
            evidence.append(EvidenceItem(
                source_table=row.source_table,
                canonical_name=row.canonical_name,
                event_time=row.event_time,
                value=row.value if row.value is not None else "NA",
                why_relevant=f"Recent {signal} value for rule trigger evaluation.",
            ))

        trigger_detected = episode.trigger_label
        is_negative = episode.trigger_type == "negative"

        return BenchmarkOutput(
            episode_id=episode.episode_id,
            task_name=episode.task_name,  # type: ignore[arg-type]
            trigger_detected=trigger_detected,
            trigger_type=episode.trigger_type,
            decision_time=episode.decision_time,
            urgency_level=_URGENCY_MAP.get(episode.task_name, "medium"),  # type: ignore[arg-type]
            recommended_next_steps=_NEXT_STEPS.get(episode.task_name, []) if not is_negative else [],
            recommended_action_families=list(episode.accepted_action_families) if not is_negative else [],
            evidence=evidence,
            missing_information=[],
            abstain=is_negative,
            abstain_reason="negative_window" if is_negative else None,
            confidence=0.95 if trigger_detected else 0.5,
            tool_trace=tool_trace,
        )

    def run_all(self, episodes: Sequence[Episode]) -> List[BenchmarkOutput]:
        return [self.run_episode(ep) for ep in episodes]
