"""Deterministic patient-level train/validation/test splitting (Phase 6).

Splits are performed at the **patient** (``subject_id``) level so that no
patient appears in more than one partition.  The assignment is deterministic
given a fixed seed and is reproducible across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42
_DEFAULT_TRAIN = 0.70
_DEFAULT_VAL = 0.15


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------

def assign_patient_splits(
    subject_ids: list[int],
    *,
    seed: int = _DEFAULT_SEED,
    train_frac: float = _DEFAULT_TRAIN,
    val_frac: float = _DEFAULT_VAL,
) -> dict[int, str]:
    """Assign each subject_id to train / val / test deterministically.

    The assignment uses a hash of ``(seed, subject_id)`` to avoid dependency
    on input ordering.  This makes splits stable regardless of how many
    patients are in the cohort.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    assignments: dict[int, str] = {}
    for sid in subject_ids:
        h = hashlib.sha256(f"{seed}|{sid}".encode()).hexdigest()
        bucket = int(h[:8], 16) / 0xFFFFFFFF
        if bucket < train_frac:
            assignments[sid] = "train"
        elif bucket < train_frac + val_frac:
            assignments[sid] = "val"
        else:
            assignments[sid] = "test"
    return assignments


# ---------------------------------------------------------------------------
# Split result container
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Outcome of splitting episodes across train/val/test."""

    split_assignments: dict[int, str]
    seed: int
    train_frac: float
    val_frac: float
    split_counts: dict[str, int] = field(default_factory=dict)
    per_task_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    episode_dfs: dict[str, pd.DataFrame] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Seed={self.seed}  train={self.train_frac}  val={self.val_frac}"]
        for split, count in sorted(self.split_counts.items()):
            lines.append(f"  {split}: {count} patients")
        for task, counts in sorted(self.per_task_counts.items()):
            lines.append(f"  {task}: {counts}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Apply splits to episode DataFrames
# ---------------------------------------------------------------------------

def apply_splits_to_episodes(
    episodes_df: pd.DataFrame,
    assignments: dict[int, str],
) -> pd.DataFrame:
    """Add a ``split`` column to episodes based on patient assignments."""
    df = episodes_df.copy()
    df["split"] = df["subject_id"].map(assignments)
    unassigned = df["split"].isna().sum()
    if unassigned > 0:
        logger.warning("%d episodes have no split assignment (unknown subject_id)", unassigned)
        df["split"] = df["split"].fillna("unassigned")
    return df


def compute_split_stats(
    episodes_df: pd.DataFrame,
    task_col: str = "task_name",
) -> dict[str, dict[str, int]]:
    """Per-task split distribution: {task: {split: count}}."""
    stats: dict[str, dict[str, int]] = {}
    for task, group in episodes_df.groupby(task_col, observed=True):
        split_counts = group["split"].value_counts().to_dict()
        stats[str(task)] = {str(k): int(v) for k, v in split_counts.items()}
    return stats


# ---------------------------------------------------------------------------
# High-level: split from a directory of episode Parquet files
# ---------------------------------------------------------------------------

def split_episodes_from_dir(
    episodes_dir: Path,
    *,
    seed: int = _DEFAULT_SEED,
    train_frac: float = _DEFAULT_TRAIN,
    val_frac: float = _DEFAULT_VAL,
) -> SplitResult:
    """Load episode Parquet files from *episodes_dir*, split, and return result."""
    pq_files = sorted(episodes_dir.glob("episodes_*.parquet"))
    if not pq_files:
        raise FileNotFoundError(f"No episode Parquet files in {episodes_dir}")

    all_dfs: list[pd.DataFrame] = []
    for pq in pq_files:
        df = pd.read_parquet(pq)
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    subject_ids = sorted(combined["subject_id"].unique().tolist())

    assignments = assign_patient_splits(
        subject_ids, seed=seed, train_frac=train_frac, val_frac=val_frac,
    )

    combined = apply_splits_to_episodes(combined, assignments)

    split_counts = {
        split: len([s for s, sp in assignments.items() if sp == split])
        for split in ("train", "val", "test")
    }
    per_task = compute_split_stats(combined)

    episode_dfs: dict[str, pd.DataFrame] = {}
    for task, group in combined.groupby("task_name", observed=True):
        episode_dfs[str(task)] = group

    return SplitResult(
        split_assignments=assignments,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        split_counts=split_counts,
        per_task_counts=per_task,
        episode_dfs=episode_dfs,
    )


def split_episodes_from_dataframe(
    episodes_df: pd.DataFrame,
    *,
    seed: int = _DEFAULT_SEED,
    train_frac: float = _DEFAULT_TRAIN,
    val_frac: float = _DEFAULT_VAL,
) -> SplitResult:
    """Split a pre-loaded episodes DataFrame."""
    subject_ids = sorted(episodes_df["subject_id"].unique().tolist())
    assignments = assign_patient_splits(
        subject_ids, seed=seed, train_frac=train_frac, val_frac=val_frac,
    )

    df = apply_splits_to_episodes(episodes_df, assignments)

    split_counts = {
        split: len([s for s, sp in assignments.items() if sp == split])
        for split in ("train", "val", "test")
    }
    per_task = compute_split_stats(df)

    episode_dfs: dict[str, pd.DataFrame] = {}
    for task, group in df.groupby("task_name", observed=True):
        episode_dfs[str(task)] = group

    return SplitResult(
        split_assignments=assignments,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        split_counts=split_counts,
        per_task_counts=per_task,
        episode_dfs=episode_dfs,
    )


# ---------------------------------------------------------------------------
# I/O: manifests and stats
# ---------------------------------------------------------------------------

def write_split_manifests(result: SplitResult, output_dir: Path) -> None:
    """Write split manifests to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": result.seed,
        "train_frac": result.train_frac,
        "val_frac": result.val_frac,
        "split_counts": result.split_counts,
        "assignments": {
            str(sid): split
            for sid, split in sorted(result.split_assignments.items())
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for task_name, df in result.episode_dfs.items():
        pq_path = output_dir / f"episodes_{task_name}_split.parquet"
        df.to_parquet(str(pq_path), index=False)

    logger.info("Split manifests written to %s", output_dir)


def write_split_stats(result: SplitResult, output_dir: Path) -> None:
    """Write human-readable and machine-readable split statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_payload = {
        "seed": result.seed,
        "train_frac": result.train_frac,
        "val_frac": result.val_frac,
        "patient_split_counts": result.split_counts,
        "per_task_episode_counts": result.per_task_counts,
    }
    stats_path = output_dir / "split_stats.json"
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")

    lines = [
        "# Split Statistics (Phase 6)",
        "",
        f"- Seed: {result.seed}",
        f"- Train fraction: {result.train_frac}",
        f"- Validation fraction: {result.val_frac}",
        f"- Test fraction: {round(1.0 - result.train_frac - result.val_frac, 4)}",
        "",
        "## Patient counts per split",
        "",
        "| Split | Patients |",
        "|-------|----------|",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {result.split_counts.get(split, 0)} |")

    lines.extend(["", "## Per-task episode counts", ""])
    for task, counts in sorted(result.per_task_counts.items()):
        lines.append(f"### {task}")
        lines.append("")
        lines.append("| Split | Episodes |")
        lines.append("|-------|----------|")
        for split in ("train", "val", "test"):
            lines.append(f"| {split} | {counts.get(split, 0)} |")
        lines.append("")

    md_path = output_dir / "split_stats.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Split stats written to %s", output_dir)
