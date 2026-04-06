"""Tests for Phase 6 — Deterministic patient-level train/val/test splitting.

Covers:
- Hash-based patient assignment determinism
- No patient overlap across splits
- Fraction approximation with realistic cohort sizes
- Split manifest and stats I/O
- Integration with episode DataFrames
- CLI wiring smoke test
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mimic_triggerbench.splitting import (
    assign_patient_splits,
    apply_splits_to_episodes,
    compute_split_stats,
    split_episodes_from_dir,
    split_episodes_from_dataframe,
    write_split_manifests,
    write_split_stats,
    SplitResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episodes_df(
    n_patients: int = 20,
    episodes_per_patient: int = 3,
    task_name: str = "hyperkalemia",
) -> pd.DataFrame:
    records = []
    for pid in range(1, n_patients + 1):
        for j in range(episodes_per_patient):
            records.append({
                "episode_id": f"ep_{pid}_{j}",
                "task_name": task_name,
                "subject_id": pid,
                "hadm_id": pid * 10,
                "stay_id": pid * 100 + j,
                "decision_time": f"2150-03-01T{8 + j:02d}:00:00",
                "context_start": "2150-03-01T00:00:00",
                "trigger_label": j == 0,
                "trigger_type": "positive" if j == 0 else "negative",
                "trigger_value": 6.5 if j == 0 else 4.0,
                "accepted_action_families": ["insulin_dextrose"] if j == 0 else [],
                "observed_action_families": [],
                "mandatory_evidence_types": ["potassium_values"],
                "split": None,
            })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Assignment determinism
# ---------------------------------------------------------------------------

class TestAssignPatientSplits:
    def test_deterministic(self):
        ids = list(range(1, 101))
        a1 = assign_patient_splits(ids, seed=42)
        a2 = assign_patient_splits(ids, seed=42)
        assert a1 == a2

    def test_different_seed_different_assignment(self):
        ids = list(range(1, 101))
        a1 = assign_patient_splits(ids, seed=42)
        a2 = assign_patient_splits(ids, seed=99)
        assert a1 != a2

    def test_no_overlap_across_splits(self):
        ids = list(range(1, 201))
        assignments = assign_patient_splits(ids, seed=42)
        train = {s for s, sp in assignments.items() if sp == "train"}
        val = {s for s, sp in assignments.items() if sp == "val"}
        test = {s for s, sp in assignments.items() if sp == "test"}
        assert train & val == set()
        assert train & test == set()
        assert val & test == set()
        assert train | val | test == set(ids)

    def test_all_three_splits_populated(self):
        ids = list(range(1, 501))
        assignments = assign_patient_splits(ids, seed=42)
        splits = set(assignments.values())
        assert splits == {"train", "val", "test"}

    def test_approximate_fractions(self):
        ids = list(range(1, 1001))
        assignments = assign_patient_splits(ids, seed=42, train_frac=0.7, val_frac=0.15)
        train_n = sum(1 for v in assignments.values() if v == "train")
        val_n = sum(1 for v in assignments.values() if v == "val")
        test_n = sum(1 for v in assignments.values() if v == "test")
        assert 600 < train_n < 800
        assert 100 < val_n < 250
        assert 50 < test_n < 250

    def test_order_independent(self):
        """Shuffling input order should not change assignments."""
        ids_sorted = list(range(1, 51))
        ids_reversed = list(reversed(ids_sorted))
        a1 = assign_patient_splits(ids_sorted, seed=42)
        a2 = assign_patient_splits(ids_reversed, seed=42)
        assert a1 == a2

    def test_invalid_fractions_rejected(self):
        with pytest.raises(ValueError, match="< 1.0"):
            assign_patient_splits([1, 2, 3], train_frac=0.8, val_frac=0.3)


# ---------------------------------------------------------------------------
# Apply to episodes
# ---------------------------------------------------------------------------

class TestApplyToEpisodes:
    def test_split_column_added(self):
        df = _make_episodes_df(n_patients=10)
        assignments = assign_patient_splits(
            sorted(df["subject_id"].unique().tolist()), seed=42,
        )
        result = apply_splits_to_episodes(df, assignments)
        assert "split" in result.columns
        assert result["split"].isna().sum() == 0

    def test_no_patient_in_multiple_splits(self):
        df = _make_episodes_df(n_patients=50)
        assignments = assign_patient_splits(
            sorted(df["subject_id"].unique().tolist()), seed=42,
        )
        result = apply_splits_to_episodes(df, assignments)
        for sid, group in result.groupby("subject_id"):
            splits = group["split"].unique()
            assert len(splits) == 1, f"Patient {sid} in multiple splits: {splits}"


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_per_task_counts(self):
        df = _make_episodes_df(n_patients=10)
        assignments = assign_patient_splits(
            sorted(df["subject_id"].unique().tolist()), seed=42,
        )
        df = apply_splits_to_episodes(df, assignments)
        stats = compute_split_stats(df)
        assert "hyperkalemia" in stats
        total = sum(stats["hyperkalemia"].values())
        assert total == len(df)


# ---------------------------------------------------------------------------
# DataFrame-based split
# ---------------------------------------------------------------------------

class TestSplitFromDataframe:
    def test_full_pipeline(self):
        df = _make_episodes_df(n_patients=30)
        result = split_episodes_from_dataframe(df, seed=42)
        assert isinstance(result, SplitResult)
        assert result.seed == 42
        assert set(result.split_counts.keys()) == {"train", "val", "test"}
        assert sum(result.split_counts.values()) == 30
        assert "hyperkalemia" in result.per_task_counts


# ---------------------------------------------------------------------------
# File-based split
# ---------------------------------------------------------------------------

class TestSplitFromDir:
    def test_round_trip(self, tmp_path: Path):
        df = _make_episodes_df(n_patients=20)
        episodes_dir = tmp_path / "episodes"
        episodes_dir.mkdir()
        df.to_parquet(str(episodes_dir / "episodes_hyperkalemia.parquet"), index=False)

        result = split_episodes_from_dir(episodes_dir, seed=42)
        assert isinstance(result, SplitResult)
        assert sum(result.split_counts.values()) == 20

    def test_no_files_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            split_episodes_from_dir(tmp_path)


# ---------------------------------------------------------------------------
# Manifest and stats I/O
# ---------------------------------------------------------------------------

class TestIO:
    def test_write_manifests(self, tmp_path: Path):
        df = _make_episodes_df(n_patients=10)
        result = split_episodes_from_dataframe(df, seed=42)
        write_split_manifests(result, tmp_path)
        assert (tmp_path / "split_manifest.json").exists()
        manifest = json.loads((tmp_path / "split_manifest.json").read_text())
        assert manifest["seed"] == 42
        assert "assignments" in manifest

    def test_write_stats(self, tmp_path: Path):
        df = _make_episodes_df(n_patients=10)
        result = split_episodes_from_dataframe(df, seed=42)
        write_split_stats(result, tmp_path)
        assert (tmp_path / "split_stats.json").exists()
        assert (tmp_path / "split_stats.md").exists()
        stats = json.loads((tmp_path / "split_stats.json").read_text())
        assert "patient_split_counts" in stats
        md = (tmp_path / "split_stats.md").read_text()
        assert "Seed: 42" in md

    def test_split_parquet_files_written(self, tmp_path: Path):
        df = _make_episodes_df(n_patients=10)
        result = split_episodes_from_dataframe(df, seed=42)
        write_split_manifests(result, tmp_path)
        assert (tmp_path / "episodes_hyperkalemia_split.parquet").exists()
        loaded = pd.read_parquet(tmp_path / "episodes_hyperkalemia_split.parquet")
        assert "split" in loaded.columns
        assert loaded["split"].isna().sum() == 0
