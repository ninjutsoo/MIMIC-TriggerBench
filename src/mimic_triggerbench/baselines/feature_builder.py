"""Deterministic feature-table builder for tabular baselines (Phase 8).

This module is the **only** way tabular baselines obtain features, ensuring
experiments are reproducible and comparable.  It enforces:

- Fixed lookback windows, bins, and aggregation rules
- Named columns with explicit types
- Missingness and measurement-count features alongside value aggregates
- Split-aware fit/transform: learned preprocessing (imputation, scaling)
  is fit on train only and reused unchanged for val/test
- Persisted ``feature_spec_version``, config hash, and split seed
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from mimic_triggerbench.labeling.episode_models import Episode
from mimic_triggerbench.timeline.models import CanonicalEvent
from mimic_triggerbench.timeline.slicing import slice_lookback, filter_by_category, filter_by_canonical_name

logger = logging.getLogger(__name__)

FEATURE_SPEC_VERSION = "v0.1"

_LOOKBACK_HOURS = 24.0

_VITAL_SIGNALS = ["map", "heart_rate", "sbp", "dbp", "resp_rate", "spo2", "temperature"]
_LAB_SIGNALS = ["potassium", "glucose", "sodium", "creatinine", "bun", "lactate", "bicarbonate", "chloride"]

_AGG_FUNCS = {
    "last": lambda vals: vals[-1] if vals else np.nan,
    "mean": lambda vals: np.mean(vals) if vals else np.nan,
    "min": lambda vals: np.min(vals) if vals else np.nan,
    "max": lambda vals: np.max(vals) if vals else np.nan,
    "std": lambda vals: np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
    "count": lambda vals: float(len(vals)),
}


def _config_hash() -> str:
    """Deterministic hash of the feature spec configuration."""
    payload = json.dumps({
        "version": FEATURE_SPEC_VERSION,
        "lookback_hours": _LOOKBACK_HOURS,
        "vital_signals": _VITAL_SIGNALS,
        "lab_signals": _LAB_SIGNALS,
        "agg_funcs": sorted(_AGG_FUNCS.keys()),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class FeatureArtifact:
    """Metadata persisted alongside feature tables for reproducibility."""
    feature_spec_version: str = FEATURE_SPEC_VERSION
    config_hash: str = field(default_factory=_config_hash)
    split_seed: Optional[int] = None
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature_spec_version": self.feature_spec_version,
            "config_hash": self.config_hash,
            "split_seed": self.split_seed,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
        }


def _extract_signal_features(
    events: List[CanonicalEvent],
    decision_time,
    signal_name: str,
    category: str,
) -> Dict[str, float]:
    """Extract aggregated features for a single signal within the lookback window."""
    sliced = slice_lookback(events, decision_time, _LOOKBACK_HOURS)
    if category == "lab":
        filtered = filter_by_category(sliced, {"lab"})
    else:
        filtered = filter_by_category(sliced, {"vital"})
    matched = filter_by_canonical_name(filtered, {signal_name})

    vals = [e.value_num for e in matched if e.value_num is not None]
    vals_sorted = sorted(
        [(e.event_time, e.value_num) for e in matched if e.value_num is not None],
        key=lambda x: x[0],
    )
    ordered_vals = [v for _, v in vals_sorted]

    features: Dict[str, float] = {}
    for agg_name, agg_fn in _AGG_FUNCS.items():
        features[f"{signal_name}_{agg_name}"] = agg_fn(ordered_vals)
    features[f"{signal_name}_missing"] = 1.0 if not ordered_vals else 0.0
    return features


class FeatureBuilder:
    """Deterministic, split-aware feature extraction for tabular baselines.

    Usage::

        fb = FeatureBuilder(timelines)
        X_train, y_train = fb.fit_transform(train_episodes)
        X_val, y_val = fb.transform(val_episodes)
        X_test, y_test = fb.transform(test_episodes)
    """

    def __init__(
        self,
        timelines: Dict[int, List[CanonicalEvent]],
        *,
        split_seed: Optional[int] = None,
    ) -> None:
        self._timelines = timelines
        self._split_seed = split_seed
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._fitted = False
        self._feature_names: List[str] = []

    @property
    def artifact(self) -> FeatureArtifact:
        return FeatureArtifact(
            split_seed=self._split_seed,
            n_features=len(self._feature_names),
            feature_names=list(self._feature_names),
        )

    def _build_raw_features(self, episodes: Sequence[Episode]) -> tuple[pd.DataFrame, np.ndarray]:
        """Extract raw (un-preprocessed) features for a list of episodes."""
        rows: List[Dict[str, float]] = []
        labels: List[int] = []

        for ep in episodes:
            stay_events = list(self._timelines.get(ep.stay_id, []))
            row: Dict[str, float] = {}
            for sig in _LAB_SIGNALS:
                row.update(_extract_signal_features(stay_events, ep.decision_time, sig, "lab"))
            for sig in _VITAL_SIGNALS:
                row.update(_extract_signal_features(stay_events, ep.decision_time, sig, "vital"))
            rows.append(row)
            labels.append(1 if ep.trigger_label else 0)

        df = pd.DataFrame(rows)
        if not self._feature_names:
            self._feature_names = list(df.columns)
        else:
            df = df.reindex(columns=self._feature_names, fill_value=np.nan)
        return df, np.array(labels)

    def fit_transform(self, episodes: Sequence[Episode]) -> tuple[np.ndarray, np.ndarray]:
        """Fit imputer+scaler on train episodes, then return transformed features + labels."""
        df, y = self._build_raw_features(episodes)
        self._imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imp = self._imputer.fit_transform(df.values)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X_imp)
        self._fitted = True
        logger.info(
            "FeatureBuilder fit on %d episodes (%d features), config_hash=%s",
            len(episodes), X.shape[1], _config_hash(),
        )
        return X, y

    def transform(self, episodes: Sequence[Episode]) -> tuple[np.ndarray, np.ndarray]:
        """Transform val/test episodes using train-fitted imputer+scaler."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first.")
        df, y = self._build_raw_features(episodes)
        X_imp = self._imputer.transform(df.values)  # type: ignore[union-attr]
        X = self._scaler.transform(X_imp)  # type: ignore[union-attr]
        return X, y
