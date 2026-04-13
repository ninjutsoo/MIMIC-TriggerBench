"""Identical scoring path for every system — baselines and agent alike (Phase 8).

Metrics:
- trigger_correct: did the system detect / not detect the trigger correctly?
- action_family_precision / recall / f1: overlap between recommended and gold action families
- abstain_correct: did the system abstain appropriately for negative episodes?
- evidence_coverage: fraction of mandatory evidence types present in evidence list
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from mimic_triggerbench.labeling.episode_models import Episode
from mimic_triggerbench.schemas.outputs import BenchmarkOutput


@dataclass(frozen=True)
class EpisodeScore:
    episode_id: str
    task_name: str
    split: Optional[str]

    trigger_correct: bool
    abstain_correct: bool

    action_family_precision: float
    action_family_recall: float
    action_family_f1: float

    evidence_coverage: float
    confidence: float


@dataclass
class RunScores:
    system_name: str
    episode_scores: List[EpisodeScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.episode_scores)

    @property
    def trigger_accuracy(self) -> float:
        if not self.episode_scores:
            return 0.0
        return sum(s.trigger_correct for s in self.episode_scores) / self.n

    @property
    def abstain_accuracy(self) -> float:
        if not self.episode_scores:
            return 0.0
        return sum(s.abstain_correct for s in self.episode_scores) / self.n

    @property
    def mean_action_f1(self) -> float:
        if not self.episode_scores:
            return 0.0
        return sum(s.action_family_f1 for s in self.episode_scores) / self.n

    @property
    def mean_evidence_coverage(self) -> float:
        if not self.episode_scores:
            return 0.0
        return sum(s.evidence_coverage for s in self.episode_scores) / self.n

    def summary_dict(self) -> Dict[str, object]:
        return {
            "system_name": self.system_name,
            "n_episodes": self.n,
            "trigger_accuracy": round(self.trigger_accuracy, 4),
            "abstain_accuracy": round(self.abstain_accuracy, 4),
            "mean_action_f1": round(self.mean_action_f1, 4),
            "mean_evidence_coverage": round(self.mean_evidence_coverage, 4),
        }


def _set_f1(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_episode(
    output: BenchmarkOutput,
    gold: Episode,
) -> EpisodeScore:
    """Score a single system output against gold episode labels."""
    trigger_correct = output.trigger_detected == gold.trigger_label

    if gold.trigger_type == "negative":
        abstain_correct = output.abstain is True
    else:
        abstain_correct = output.abstain is False

    pred_families = set(output.recommended_action_families)
    gold_families = set(gold.accepted_action_families)
    prec, rec, f1 = _set_f1(pred_families, gold_families)

    mandatory = set(gold.mandatory_evidence_types)
    if mandatory:
        found = {e.canonical_name for e in output.evidence}
        evidence_coverage = len(found & mandatory) / len(mandatory)
    else:
        evidence_coverage = 1.0

    return EpisodeScore(
        episode_id=output.episode_id,
        task_name=output.task_name,
        split=gold.split,
        trigger_correct=trigger_correct,
        abstain_correct=abstain_correct,
        action_family_precision=prec,
        action_family_recall=rec,
        action_family_f1=f1,
        evidence_coverage=evidence_coverage,
        confidence=output.confidence,
    )


def score_run(
    system_name: str,
    outputs: Sequence[BenchmarkOutput],
    golds: Sequence[Episode],
) -> RunScores:
    """Score a full run (list of outputs matched 1:1 with gold episodes)."""
    gold_map = {g.episode_id: g for g in golds}
    scores = RunScores(system_name=system_name)
    for out in outputs:
        gold = gold_map.get(out.episode_id)
        if gold is None:
            raise ValueError(f"No gold episode for output episode_id={out.episode_id}")
        scores.episode_scores.append(score_episode(out, gold))
    return scores


def aggregate_scores(
    run_scores: RunScores,
    by: str = "task_name",
) -> Dict[str, Dict[str, float]]:
    """Group episode scores by *by* field (task_name or split) and compute per-group means."""
    groups: Dict[str, list[EpisodeScore]] = {}
    for s in run_scores.episode_scores:
        key = getattr(s, by, "all")
        groups.setdefault(key, []).append(s)

    result: Dict[str, Dict[str, float]] = {}
    for key, scores in sorted(groups.items()):
        n = len(scores)
        result[key] = {
            "n": n,
            "trigger_accuracy": round(sum(s.trigger_correct for s in scores) / n, 4),
            "abstain_accuracy": round(sum(s.abstain_correct for s in scores) / n, 4),
            "mean_action_f1": round(sum(s.action_family_f1 for s in scores) / n, 4),
            "mean_evidence_coverage": round(sum(s.evidence_coverage for s in scores) / n, 4),
        }
    return result
