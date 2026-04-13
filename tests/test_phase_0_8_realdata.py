"""Real-data integration test for implemented phases 0-8.

Extends the Phase 0-7 real-data harness to exercise Phase 8 baselines
and the evaluation harness on actual MIMIC data.  Covers:

- Rule baseline on all three task families
- Tabular ML baseline (train-only fit, val/test transform)
- LLM + RAG baselines (local HF on one GPU when ``TRIGGERBENCH_RUN_HF_TESTS=1``; otherwise skipped in summary)
- Evaluation scoring for every baseline
- Per-baseline timing and HF generation counts (when local LLM enabled)

Skip: when no local MIMIC data is available.

Inspectability artifacts:
- ``tmp_path/phase_0_8_realdata_run.json``: full run summary
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from mimic_triggerbench.config import DataBackend, Settings, load_settings
from mimic_triggerbench.labeling import (
    Episode,
    episodes_to_records,
    generate_all_episodes,
    load_all_task_specs,
)
from mimic_triggerbench.mimic_tables import iter_table_specs, resolve_table_path
from mimic_triggerbench.data_access.tables import load_table_dataframe
from mimic_triggerbench.replay import ReplayEnvironment
from mimic_triggerbench.schemas import validate_benchmark_output
from mimic_triggerbench.splitting import split_episodes_from_dataframe
from mimic_triggerbench.timeline import build_all_timelines

from mimic_triggerbench.baselines.rule_baseline import RuleBaseline
from mimic_triggerbench.baselines.tabular_baseline import TabularBaseline
from mimic_triggerbench.baselines.feature_builder import FeatureBuilder
from mimic_triggerbench.evaluation.scoring import score_run, aggregate_scores, RunScores

KNOWN_TASK_NAMES = frozenset({"hyperkalemia", "hypoglycemia", "hypotension"})
LOGGER = logging.getLogger(__name__)

MAX_EPISODES_PER_TASK = 20


# ---------------------------------------------------------------------------
# Harness utilities
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _step(name: str, summary: dict):
    t0 = time.perf_counter()
    LOGGER.info("[phase_0_8_realdata] START %s", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        LOGGER.info("[phase_0_8_realdata] DONE  %s (%.2fs)", name, dt)
        summary.setdefault("timings_sec", {})[name] = round(dt, 4)


def _run_with_timeout(name: str, timeout_s: int, fn, *, tmp_path: Path):
    q: queue.Queue = queue.Queue(maxsize=1)

    def _runner():
        try:
            q.put(("ok", fn()))
        except BaseException as e:
            q.put(("err", e))

    t = threading.Thread(target=_runner, name=f"phase_0_8::{name}", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        import faulthandler
        dump_path = tmp_path / "phase_0_8_realdata_stuck_stacks.txt"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w") as f:
            faulthandler.dump_traceback(file=f, all_threads=True)
        pytest.fail(f"Step {name!r} exceeded timeout ({timeout_s}s).")
    status, payload = q.get_nowait()
    if status == "err":
        raise payload
    return payload


# ---------------------------------------------------------------------------
# MIMIC data discovery (shared with test_phase_0_7_realdata)
# ---------------------------------------------------------------------------


def _has_required_files(mimic_root: Path) -> bool:
    return all(
        resolve_table_path(mimic_root, spec.table_name) is not None
        for spec in iter_table_specs()
    )


def _discover_repo_mimic_root() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / "physionet.org" / "files" / "mimiciv"
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    for version_dir in sorted(candidates, key=lambda p: p.name, reverse=True):
        if _has_required_files(version_dir):
            return version_dir
    return None


def _load_real_settings() -> Settings | None:
    try:
        settings = load_settings(dotenv_path=None)
    except Exception:
        settings = None
    if (
        settings is not None
        and settings.backend == DataBackend.FILES
        and settings.mimic_root
        and _has_required_files(settings.mimic_root)
    ):
        return settings
    discovered = _discover_repo_mimic_root()
    if discovered is not None:
        return Settings(backend=DataBackend.FILES, mimic_root=discovered)
    return None


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def test_implemented_phases_0_8_realdata_end_to_end(tmp_path: Path) -> None:
    """End-to-end real-data test for phases 0-8 (baselines + evaluation)."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_summary: dict = {}
    step_timeout_s = 300

    settings = _load_real_settings()
    if settings is None or settings.mimic_root is None:
        pytest.skip("No local MIMIC data available.")
    if not _has_required_files(settings.mimic_root):
        pytest.skip(f"MIMIC files not found under {settings.mimic_root!s}.")

    mimic_root = Path(settings.mimic_root)
    run_summary["mimic_root"] = str(mimic_root)

    # =================================================================
    # Phase 0-4: load tables + build timelines
    # =================================================================
    with _step("load_icustays", run_summary):
        try:
            icu_df = load_table_dataframe(settings, "icustays")
        except Exception as e:
            pytest.skip(f"Could not read icustays: {e!r}")
    if icu_df.empty:
        pytest.skip("icustays table is empty.")

    max_rows_per_table = 50_000
    max_stays = 200

    with _step("build_timelines", run_summary):
        result = _run_with_timeout(
            "build_timelines", step_timeout_s,
            lambda: build_all_timelines(
                settings,
                max_rows_per_table=max_rows_per_table,
                max_stays=max_stays,
            ),
            tmp_path=tmp_path,
        )
    timelines, tl_stats = result
    assert len(timelines) > 0
    run_summary["timeline_stays"] = len(timelines)
    run_summary["timeline_total_events"] = tl_stats.total_events
    run_summary["max_rows_per_table"] = max_rows_per_table
    run_summary["max_stays"] = max_stays

    # =================================================================
    # Phase 5: episode generation
    # =================================================================
    specs = load_all_task_specs()
    all_episodes: List[Episode] = []

    with _step("episode_generation", run_summary):
        for task_name, spec in specs.items():
            task_episodes = _run_with_timeout(
                f"episodes_{task_name}", step_timeout_s,
                lambda _s=spec: generate_all_episodes(_s, timelines),
                tmp_path=tmp_path,
            )
            all_episodes.extend(task_episodes)

    assert len(all_episodes) > 0
    run_summary["total_episodes"] = len(all_episodes)
    run_summary["episodes_per_task"] = {
        task: sum(1 for e in all_episodes if e.task_name == task)
        for task in KNOWN_TASK_NAMES
    }

    # =================================================================
    # Phase 6: splitting
    # =================================================================
    records = episodes_to_records(all_episodes)
    episodes_df = pd.DataFrame.from_records(records)

    with _step("splitting", run_summary):
        split_result = split_episodes_from_dataframe(episodes_df, seed=42)

    assigned_df = episodes_df.copy()
    assigned_df["split"] = assigned_df["subject_id"].map(split_result.split_assignments)

    episode_map: Dict[str, Episode] = {e.episode_id: e for e in all_episodes}
    split_episodes: Dict[str, List[Episode]] = {"train": [], "val": [], "test": []}
    for _, row in assigned_df.iterrows():
        ep = episode_map[row["episode_id"]]
        sp = row["split"]
        if sp in split_episodes:
            ep_with_split = ep.model_copy(update={"split": sp})
            split_episodes[sp].append(ep_with_split)

    for sp, eps in split_episodes.items():
        run_summary[f"{sp}_episodes"] = len(eps)

    train_eps = split_episodes["train"]
    val_eps = split_episodes["val"]
    test_eps = split_episodes["test"]

    def _cap_per_task(eps: List[Episode], cap: int) -> List[Episode]:
        """Cap to at most *cap* episodes per task, preserving task diversity."""
        by_task: Dict[str, List[Episode]] = {}
        for e in eps:
            by_task.setdefault(e.task_name, []).append(e)
        out: List[Episode] = []
        for task in sorted(by_task):
            out.extend(by_task[task][:cap])
        return out

    capped_train = _cap_per_task(train_eps, MAX_EPISODES_PER_TASK)
    capped_val = _cap_per_task(val_eps, MAX_EPISODES_PER_TASK)
    capped_test = _cap_per_task(test_eps, MAX_EPISODES_PER_TASK)

    run_summary["capped_train"] = len(capped_train)
    run_summary["capped_val"] = len(capped_val)
    run_summary["capped_test"] = len(capped_test)

    # =================================================================
    # Phase 7: replay environment
    # =================================================================
    env = ReplayEnvironment(timelines)

    # =================================================================
    # Phase 8A: Rule baseline
    # =================================================================
    with _step("rule_baseline", run_summary):
        rule_bl = RuleBaseline(env, specs)
        rule_outputs = rule_bl.run_all(capped_val)

    for out in rule_outputs:
        validate_benchmark_output(out.model_dump(mode="json"))

    with _step("rule_baseline_scoring", run_summary):
        rule_scores = score_run("rule_baseline", rule_outputs, capped_val)
    run_summary["rule_baseline_scores"] = rule_scores.summary_dict()
    run_summary["rule_baseline_by_task"] = aggregate_scores(rule_scores, by="task_name")

    # =================================================================
    # Phase 8B: Tabular ML baseline (train-only fit)
    # =================================================================
    with _step("tabular_feature_build_train", run_summary):
        fb = FeatureBuilder(timelines, split_seed=42)
        fb.fit_transform(capped_train)

    with _step("tabular_feature_transform_val", run_summary):
        fb.transform(capped_val)

    run_summary["tabular_feature_artifact"] = fb.artifact.to_dict()

    tabular_outputs = []
    if len(capped_train) >= 4:
        with _step("tabular_fit", run_summary):
            tb = TabularBaseline(fb, model_type="xgboost", seed=42)
            tb.fit(capped_train)

        with _step("tabular_predict", run_summary):
            tabular_outputs = tb.predict(capped_val)

        for out in tabular_outputs:
            validate_benchmark_output(out.model_dump(mode="json"))

        with _step("tabular_scoring", run_summary):
            tabular_scores = score_run("tabular_baseline", tabular_outputs, capped_val)
        run_summary["tabular_baseline_scores"] = tabular_scores.summary_dict()
        run_summary["tabular_baseline_by_task"] = aggregate_scores(tabular_scores, by="task_name")
    else:
        run_summary["tabular_baseline_scores"] = {"skipped": "insufficient training data (<4 episodes)"}

    # =================================================================
    # Phase 8C–D: LLM + RAG baselines (local Hugging Face, opt-in)
    # =================================================================
    run_hf = os.environ.get("TRIGGERBENCH_RUN_HF_TESTS") == "1"
    if run_hf:
        try:
            import torch
        except ImportError as e:
            pytest.fail(f"TRIGGERBENCH_RUN_HF_TESTS=1 requires torch (pip install -e '.[llm]'): {e}")
        if not torch.cuda.is_available():
            pytest.fail("TRIGGERBENCH_RUN_HF_TESTS=1 requires CUDA for local HF baselines.")

    from mimic_triggerbench.baselines.hf_local_llm import get_default_model_id

    run_summary["hf_model_id"] = get_default_model_id()
    run_summary["hf_home"] = os.environ.get("HF_HOME", "")

    llm_capped = capped_val[:3] if run_hf else []
    run_summary["llm_baseline_attempted"] = run_hf
    run_summary["llm_baseline_episodes"] = len(llm_capped)

    if llm_capped:
        from mimic_triggerbench.baselines.hf_local_llm import reset_shared_local_generator_for_tests
        from mimic_triggerbench.baselines.llm_baseline import LLMBaseline

        reset_shared_local_generator_for_tests()
        with _step("llm_baseline_total", run_summary):
            llm_bl = LLMBaseline(env, specs)
            llm_outputs = llm_bl.run_all(llm_capped)

        for out in llm_outputs:
            validate_benchmark_output(out.model_dump(mode="json"))

        with _step("llm_scoring", run_summary):
            llm_scores = score_run("llm_baseline", llm_outputs, llm_capped)
        run_summary["llm_baseline_scores"] = llm_scores.summary_dict()
        run_summary["hf_generations_llm"] = llm_bl.generation_count
    else:
        run_summary["llm_baseline_scores"] = {"skipped": "TRIGGERBENCH_RUN_HF_TESTS not set to 1"}
        run_summary["hf_generations_llm"] = 0

    rag_capped = capped_val[:3] if run_hf else []
    run_summary["rag_baseline_attempted"] = run_hf
    run_summary["rag_baseline_episodes"] = len(rag_capped)

    if rag_capped:
        from mimic_triggerbench.baselines.rag_baseline import RAGBaseline

        with _step("rag_baseline_total", run_summary):
            rag_bl = RAGBaseline(env, specs)
            rag_outputs = rag_bl.run_all(rag_capped)

        for out in rag_outputs:
            validate_benchmark_output(out.model_dump(mode="json"))

        with _step("rag_scoring", run_summary):
            rag_scores = score_run("rag_baseline", rag_outputs, rag_capped)
        run_summary["rag_baseline_scores"] = rag_scores.summary_dict()
        run_summary["hf_generations_rag"] = rag_bl.generation_count
    else:
        run_summary["rag_baseline_scores"] = {"skipped": "TRIGGERBENCH_RUN_HF_TESTS not set to 1"}
        run_summary["hf_generations_rag"] = 0

    run_summary["hf_generations_total"] = (
        run_summary.get("hf_generations_llm", 0) + run_summary.get("hf_generations_rag", 0)
    )

    if run_hf:
        from mimic_triggerbench.baselines.hf_local_llm import reset_shared_local_generator_for_tests

        reset_shared_local_generator_for_tests()

    # =================================================================
    # Write run summary
    # =================================================================
    summary_path = tmp_path / "phase_0_8_realdata_run.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, default=str), encoding="utf-8")
    assert summary_path.exists()

    LOGGER.info("Phase 0-8 real-data run summary written to %s", summary_path)
    LOGGER.info("Timings: %s", json.dumps(run_summary.get("timings_sec", {}), indent=2))
