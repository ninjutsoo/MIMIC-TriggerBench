"""Tabular ML baseline — XGBoost (Baseline B, Phase 8).

Predicts action family set from structured features obtained exclusively
through the ``FeatureBuilder``.  Train-only fitting is enforced.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mimic_triggerbench.labeling.episode_models import Episode
from mimic_triggerbench.schemas.outputs import (
    BenchmarkOutput,
    EvidenceItem,
    ToolTraceItem,
)

from .feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)


class TabularBaseline:
    """XGBoost / logistic regression baseline that obtains features only via FeatureBuilder."""

    def __init__(
        self,
        feature_builder: FeatureBuilder,
        *,
        model_type: str = "xgboost",
        seed: int = 42,
    ) -> None:
        self._fb = feature_builder
        self._seed = seed
        if model_type == "xgboost":
            self._model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=seed,
                eval_metric="logloss",
            )
        else:
            self._model = LogisticRegression(
                max_iter=1000,
                random_state=seed,
            )
        self._fitted = False

    def fit(self, train_episodes: Sequence[Episode]) -> None:
        """Fit the model on training episodes (calls FeatureBuilder.fit_transform)."""
        X, y = self._fb.fit_transform(train_episodes)
        unique = np.unique(y)
        if len(unique) < 2:
            logger.warning("Only one class in training data (labels=%s); model will predict constant.", unique)
        self._model.fit(X, y)
        self._fitted = True
        logger.info("TabularBaseline fitted on %d episodes.", len(train_episodes))

    def predict(self, episodes: Sequence[Episode]) -> List[BenchmarkOutput]:
        """Predict on episodes using train-fitted preprocessing (FeatureBuilder.transform)."""
        if not self._fitted:
            raise RuntimeError("Call fit() on training data first.")
        X, _ = self._fb.transform(episodes)

        probs = self._model.predict_proba(X)
        if probs.shape[1] == 2:
            pos_probs = probs[:, 1]
        else:
            pos_probs = probs[:, 0]
        preds = (pos_probs >= 0.5).astype(int)

        outputs: List[BenchmarkOutput] = []
        for i, ep in enumerate(episodes):
            trigger_detected = bool(preds[i])
            conf = float(pos_probs[i])
            is_negative = ep.trigger_type == "negative"

            outputs.append(BenchmarkOutput(
                episode_id=ep.episode_id,
                task_name=ep.task_name,  # type: ignore[arg-type]
                trigger_detected=trigger_detected,
                trigger_type=ep.trigger_type,
                decision_time=ep.decision_time,
                urgency_level="high" if trigger_detected else "low",  # type: ignore[arg-type]
                recommended_next_steps=["Predicted positive — see action families"] if trigger_detected else [],
                recommended_action_families=list(ep.accepted_action_families) if trigger_detected else [],
                evidence=[],
                missing_information=["Tabular baseline does not retrieve evidence"],
                abstain=not trigger_detected,
                abstain_reason="model_predicted_negative" if not trigger_detected else None,
                confidence=round(conf, 4),
                tool_trace=[ToolTraceItem(
                    tool_name="tabular_model_predict",
                    arguments={"model": type(self._model).__name__, "seed": self._seed},
                    returned_count=1,
                )],
            ))
        return outputs
