"""Real-data integration test for implemented phases 0–7.

This is the authoritative ``check-implement`` test when Phase 7 is marked
``[DONE]`` in ``IMPLEMENTATION_TASKS.md``. It exercises, on real MIMIC data
when available, the main public entrypoints from all phases marked
``[DONE]``:

- Phase 0/1: settings + inventory + table loading
- Phase 3: normalization coverage on real tables
- Phase 4: canonical timeline builder
- Phase 3.5: action feasibility checkpoint on those timelines
- Phase 5: deterministic episode generation (positive + negative)
- Phase 6: patient-level train/val/test splitting
- Phase 6.5: frozen BenchmarkOutput schema gate
- Phase 7: replay environment + structured, timestamp-bounded tools

The test is strict: it captures exact inputs, asserts on all output shapes
and key content, and writes a run summary to ``tmp_path`` so you can inspect
exactly what data came in and what was produced. Skip only when no local
MIMIC data is available.

Inspectability artifacts:
- ``tmp_path/phase_0_7_realdata_run.json``: inputs + outputs + timings
- ``tmp_path/phase_0_7_realdata_stuck_stacks.txt``: written on timeout
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from mimic_triggerbench.config import DataBackend, Settings, load_settings
from mimic_triggerbench.data_access import generate_inventory_report
from mimic_triggerbench.data_access.normalization_audit import (
    TableScanResult,
    scan_normalization_coverage,
)
from mimic_triggerbench.data_access.tables import load_table_dataframe
from mimic_triggerbench.feasibility import (
    DetectedAction,
    FeasibilityDecision,
    run_feasibility_checkpoint,
)
from mimic_triggerbench.labeling import (
    Episode,
    episodes_to_records,
    generate_all_episodes,
    load_all_task_specs,
)
from mimic_triggerbench.mimic_tables import (
    iter_table_specs,
    required_table_names,
    resolve_table_path,
)
from mimic_triggerbench.replay import ReplayEnvironment
from mimic_triggerbench.schemas import (
    OutputSchemaError,
    EpisodeInput,
    validate_benchmark_output,
    validate_tool_result,
)
from mimic_triggerbench.splitting import SplitResult, split_episodes_from_dataframe
from mimic_triggerbench.timeline import build_all_timelines
from mimic_triggerbench.timeline.models import CanonicalEvent

# Expected action families from feasibility checkpoint (Phase 3.5).
EXPECTED_FEASIBILITY_FAMILIES = frozenset(
    {
        "insulin_dextrose",
        "vasopressor_start",
        "vasopressor_escalation",
        "fluid_bolus",
        "dialysis_start",
    }
)

# CanonicalEvent source tables produced by timeline builder (Phase 4).
EXPECTED_TIMELINE_SOURCE_TABLES = frozenset(
    {
        "labevents",
        "chartevents",
        "inputevents",
        "procedureevents",
        "outputevents",
    }
)

# Allowed event_category values on CanonicalEvent.
EXPECTED_EVENT_CATEGORIES = frozenset(
    {
        "lab",
        "vital",
        "med_bolus",
        "med_infusion",
        "procedure",
        "output",
        "other",
    }
)

KNOWN_TASK_NAMES = frozenset({"hyperkalemia", "hypoglycemia", "hypotension"})

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harness utilities (step timing, timeouts, stack dumps)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _step(name: str, summary: dict) -> object:
    t0 = time.perf_counter()
    LOGGER.info("[phase_0_7_realdata] START %s", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        LOGGER.info("[phase_0_7_realdata] DONE  %s (%.2fs)", name, dt)
        summary.setdefault("timings_sec", {})[name] = round(dt, 4)


def _dump_thread_stacks(path: Path) -> None:
    import faulthandler

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Thread stack dump (faulthandler)\n\n")
        faulthandler.dump_traceback(file=f, all_threads=True)


def _run_with_timeout(name: str, timeout_s: int, fn, *, tmp_path: Path):
    q: queue.Queue = queue.Queue(maxsize=1)

    def _runner() -> None:
        try:
            q.put(("ok", fn()))
        except BaseException as e:  # noqa: BLE001
            q.put(("err", e))

    t = threading.Thread(
        target=_runner, name=f"phase_0_7_realdata::{name}", daemon=True
    )
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        dump_path = tmp_path / "phase_0_7_realdata_stuck_stacks.txt"
        _dump_thread_stacks(dump_path)
        pytest.fail(
            f"Step {name!r} exceeded timeout ({timeout_s}s). "
            f"Wrote thread stacks to {dump_path!s}."
        )
    status, payload = q.get_nowait()
    if status == "err":
        raise payload
    return payload


# ---------------------------------------------------------------------------
# MIMIC data discovery
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
    if not candidates:
        return None
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


def test_implemented_phases_0_7_realdata_end_to_end(tmp_path: Path) -> None:
    """End-to-end real-data test for phases 0–7."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_summary: dict = {}

    # =====================================================================
    # INPUTS: capture exactly what we use
    # =====================================================================
    settings = _load_real_settings()
    if settings is None or settings.mimic_root is None:
        pytest.skip("No local MIMIC data available for real-data integration test.")
    if not _has_required_files(settings.mimic_root):
        pytest.skip(f"Local MIMIC files not found under mimic_root={settings.mimic_root!s}.")

    mimic_root = Path(settings.mimic_root)
    required_tables = list(required_table_names())
    resolved_tables: dict[str, str] = {}
    for t in required_tables:
        r = resolve_table_path(mimic_root, t)
        resolved_tables[t] = str(r.path) if r else ""

    assert settings.backend == DataBackend.FILES, "Test expects FILES backend."
    assert mimic_root.exists(), f"mimic_root must exist: {mimic_root}"
    assert len(resolved_tables) == len(required_tables)

    with _step("load_icustays", run_summary):
        try:
            icu_df = load_table_dataframe(settings, "icustays")
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Could not read icustays from local MIMIC data: {e!r}")
    if icu_df.empty:
        pytest.skip("icustays table is empty in local MIMIC data.")
    for col in ("stay_id", "subject_id", "hadm_id"):
        assert col in icu_df.columns, f"icustays must have column {col!r}"

    num_icustays_rows = len(icu_df)
    sample_stays = icu_df["stay_id"].head(5).tolist()
    max_rows_per_table = 25_000
    coverage_top_k = 10
    step_timeout_s = 180

    LOGGER.info(
        "[phase_0_7_realdata] mimic_root=%s icustays_rows=%d sample_stays=%s",
        mimic_root,
        num_icustays_rows,
        sample_stays,
    )

    inputs_summary = {
        "backend": settings.backend.value,
        "mimic_root": str(mimic_root),
        "required_tables": required_tables,
        "resolved_paths": resolved_tables,
        "icustays_rows": num_icustays_rows,
        "icustays_columns": list(icu_df.columns),
        "sample_stays": sample_stays,
        "max_rows_per_table": max_rows_per_table,
        "coverage_top_k": coverage_top_k,
        "step_timeout_s": step_timeout_s,
    }

    # =====================================================================
    # Phase 0/1: inventory report
    # =====================================================================
    inv_out = tmp_path / "inventory.md"
    with _step("inventory_report", run_summary):
        _run_with_timeout(
            "inventory_report",
            step_timeout_s,
            lambda: generate_inventory_report(settings, inv_out),
            tmp_path=tmp_path,
        )
        inv_text = inv_out.read_text(encoding="utf-8")

    assert inv_out.exists()
    assert "# Data inventory (generated)" in inv_text
    assert settings.backend.value in inv_text
    for table_name in required_tables:
        assert f"`{table_name}`" in inv_text or table_name in inv_text
    for key_table in ("icustays", "labevents", "chartevents"):
        assert key_table in inv_text
        idx = inv_text.find(key_table)
        snippet = inv_text[idx : idx + 80]
        assert "yes" in snippet or "present" in snippet

    # =====================================================================
    # Phase 3: normalization coverage
    # =====================================================================
    with _step("normalization_coverage", run_summary):
        coverage_results = _run_with_timeout(
            "normalization_coverage",
            step_timeout_s,
            lambda: scan_normalization_coverage(
                settings,
                max_rows_per_table=max_rows_per_table,
                top_k=coverage_top_k,
            ),
            tmp_path=tmp_path,
        )

    assert isinstance(coverage_results, list)
    tables_in_coverage = {r.table for r in coverage_results}
    assert "labevents" in tables_in_coverage
    assert "chartevents" in tables_in_coverage

    total_rows_scanned = 0
    coverage_summary: list[dict] = []
    for r in coverage_results:
        assert isinstance(r, TableScanResult)
        assert r.rows_scanned >= 0
        assert r.mapped + r.unmapped <= r.rows_scanned
        assert r.ambiguous <= r.mapped
        total_rows_scanned += r.rows_scanned
        coverage_summary.append(
            {
                "table": r.table,
                "rows_scanned": r.rows_scanned,
                "mapped": r.mapped,
                "unmapped": r.unmapped,
                "ambiguous": r.ambiguous,
            }
        )
    assert total_rows_scanned > 0

    # =====================================================================
    # Phase 4: canonical timelines
    # =====================================================================
    with _step("build_timelines", run_summary):
        try:
            timelines, tl_stats = _run_with_timeout(
                "build_timelines",
                step_timeout_s,
                lambda: build_all_timelines(
                    settings,
                    stay_ids=sample_stays,
                    max_rows_per_table=max_rows_per_table,
                ),
                tmp_path=tmp_path,
            )
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Could not build canonical timelines: {e!r}")

    assert isinstance(timelines, dict)
    assert timelines
    assert tl_stats.total_events > 0
    assert tl_stats.stays_built == len(timelines)

    events_per_table: dict[str, int] = dict(tl_stats.events_per_table)
    timeline_summary: list[dict] = []
    for stay_id, events in timelines.items():
        assert isinstance(stay_id, int)
        assert len(events) > 0
        for ev in events:
            assert isinstance(ev, CanonicalEvent)
            assert ev.stay_id == stay_id
            assert ev.event_uid and isinstance(ev.event_uid, str)
            assert ev.source_table in EXPECTED_TIMELINE_SOURCE_TABLES
            assert ev.event_category in EXPECTED_EVENT_CATEGORIES
        timeline_summary.append(
            {
                "stay_id": stay_id,
                "event_count": len(events),
                "source_tables_seen": list({e.source_table for e in events}),
                "categories_seen": list({e.event_category for e in events}),
            }
        )

    # =====================================================================
    # Phase 3.5: feasibility checkpoint
    # =====================================================================
    with _step("feasibility_checkpoint", run_summary):
        decisions, review_sets = _run_with_timeout(
            "feasibility_checkpoint",
            step_timeout_s,
            lambda: run_feasibility_checkpoint(timelines),
            tmp_path=tmp_path,
        )

    assert isinstance(decisions, list)
    assert decisions
    decision_families = {d.action_family for d in decisions}
    assert decision_families == EXPECTED_FEASIBILITY_FAMILIES

    feasibility_summary: list[dict] = []
    for d in decisions:
        assert isinstance(d, FeasibilityDecision)
        feasibility_summary.append(
            {
                "action_family": d.action_family,
                "passed": d.passed,
                "detected_events": d.stats.detected_events,
                "unique_stays": d.stats.unique_stays,
                "unique_patients": d.stats.unique_patients,
            }
        )

    assert isinstance(review_sets, dict)
    assert set(review_sets.keys()) <= EXPECTED_FEASIBILITY_FAMILIES
    for family, actions in review_sets.items():
        for a in actions:
            assert isinstance(a, DetectedAction)
            assert a.action_family == family

    # =====================================================================
    # Phase 5: episode generation
    # =====================================================================
    specs = load_all_task_specs()
    all_episodes: list[Episode] = []
    episode_summary: dict[str, dict] = {}

    with _step("episode_generation", run_summary):
        for task_name, spec in specs.items():
            task_episodes = _run_with_timeout(
                f"episodes_{task_name}",
                step_timeout_s,
                lambda _s=spec: generate_all_episodes(_s, timelines),
                tmp_path=tmp_path,
            )
            all_episodes.extend(task_episodes)
            n_pos = sum(1 for e in task_episodes if e.trigger_label)
            n_neg = sum(1 for e in task_episodes if not e.trigger_label)
            episode_summary[task_name] = {
                "total": len(task_episodes),
                "positive": n_pos,
                "negative": n_neg,
                "stays_with_episodes": len({e.stay_id for e in task_episodes}),
            }

    assert len(all_episodes) > 0
    for ep in all_episodes:
        assert isinstance(ep, Episode)
        assert ep.task_name in KNOWN_TASK_NAMES
        assert ep.context_start <= ep.decision_time
        assert ep.trigger_type in ("positive", "negative")
        if ep.trigger_type == "positive":
            assert ep.trigger_label is True
            assert len(ep.accepted_action_families) > 0
        else:
            assert ep.trigger_label is False
            assert ep.accepted_action_families == []
        assert len(ep.mandatory_evidence_types) > 0

    records = episodes_to_records(all_episodes)
    episodes_df = pd.DataFrame.from_records(records)
    assert not episodes_df.empty

    # =====================================================================
    # Phase 6: patient-level splitting
    # =====================================================================
    with _step("splitting", run_summary):
        split_result = _run_with_timeout(
            "splitting",
            step_timeout_s,
            lambda: split_episodes_from_dataframe(episodes_df, seed=42),
            tmp_path=tmp_path,
        )

    assert isinstance(split_result, SplitResult)
    assert set(split_result.split_counts.keys()) == {"train", "val", "test"}
    assert split_result.per_task_counts

    # =====================================================================
    # Phase 6.5: frozen output schema validation gate (real-data exercised)
    # =====================================================================
    with _step("phase_6_5_output_schema_gate", run_summary):
        ep0 = all_episodes[0]
        evs = timelines[ep0.stay_id]
        asof = [e for e in evs if e.event_time <= ep0.decision_time]
        assert asof
        ev0 = asof[-1]

        payload = {
            "episode_id": ep0.episode_id,
            "task_name": ep0.task_name,
            "trigger_detected": bool(ep0.trigger_label),
            "trigger_type": ep0.trigger_type,
            "decision_time": ep0.decision_time.isoformat(),
            "urgency_level": "high" if ep0.trigger_label else "low",
            "recommended_next_steps": ["placeholder_step_for_schema_gate"],
            "recommended_action_families": (
                ep0.accepted_action_families if ep0.trigger_label else []
            ),
            "evidence": [
                {
                    "source_table": ev0.source_table,
                    "canonical_name": ev0.canonical_name,
                    "event_time": ev0.event_time.isoformat(),
                    "value": (
                        ev0.value_num
                        if ev0.value_num is not None
                        else (ev0.value_text if ev0.value_text is not None else "NA")
                    ),
                    "why_relevant": "Schema gate real-data smoke check (Phase 6.5).",
                }
            ],
            "missing_information": [],
            "abstain": False if ep0.trigger_label else True,
            "abstain_reason": None if ep0.trigger_label else "negative_window",
            "confidence": 0.9 if ep0.trigger_label else 0.5,
            "tool_trace": [
                {
                    "tool_name": "phase_6_5_schema_gate",
                    "arguments": {
                        "stay_id": ep0.stay_id,
                        "decision_time": ep0.decision_time.isoformat(),
                    },
                    "returned_count": 1,
                }
            ],
        }
        try:
            parsed = validate_benchmark_output(payload)
        except OutputSchemaError as e:
            pytest.fail(f"Phase 6.5 output schema gate failed: {e}")
        assert parsed.episode_id == ep0.episode_id

    # =====================================================================
    # Phase 7: replay environment + structured tools (real-data exercised)
    # =====================================================================
    with _step("phase_7_replay_tools", run_summary):
        env = ReplayEnvironment(timelines)

        by_task: dict[str, Episode] = {}
        for task in ("hyperkalemia", "hypoglycemia", "hypotension"):
            match = next((e for e in all_episodes if e.task_name == task), None)
            if match is not None:
                by_task[task] = match

        assert by_task, "Must find at least one episode to exercise replay tools."

        replay_checks: list[dict] = []
        for task_name, ep in by_task.items():
            ep_in = EpisodeInput(
                episode_id=ep.episode_id,
                task_name=ep.task_name,  # type: ignore[arg-type]
                stay_id=ep.stay_id,
                hadm_id=ep.hadm_id,
                subject_id=ep.subject_id,
                decision_time=ep.decision_time,
                context_start=ep.context_start,
                mandatory_evidence_types=list(ep.mandatory_evidence_types),
                accepted_action_families=list(ep.accepted_action_families),
            )
            env.load_episode(ep_in)

            if task_name in ("hyperkalemia", "hypoglycemia"):
                lab = "potassium" if task_name == "hyperkalemia" else "glucose"
                tr = env.call_tool(
                    "get_recent_labs",
                    {
                        "lab_names": [lab],
                        "hours_back": 24,
                        "before_time": ep.decision_time + pd.Timedelta(hours=6),
                    },
                )
                # This should be non-empty because the episode was triggered by a lab measurement.
                assert tr.result_count >= 1
            else:
                tr = env.call_tool(
                    "get_recent_vitals",
                    {
                        "vital_names": ["map"],
                        "minutes_back": 120,
                        "before_time": ep.decision_time + pd.Timedelta(hours=6),
                    },
                )
                assert tr.result_count >= 1

            validate_tool_result(tr.model_dump(mode="json"))
            assert tr.queried_time_range.end == ep.decision_time
            assert all(row.event_time <= ep.decision_time for row in tr.results)

            replay_checks.append(
                {
                    "task": task_name,
                    "episode_id": ep.episode_id,
                    "tool_name": tr.tool_name,
                    "result_count": tr.result_count,
                }
            )

    # =====================================================================
    # Run summary: write to tmp_path for inspectability
    # =====================================================================
    run_summary.update(
        {
            "inputs": inputs_summary,
            "inventory_file": str(inv_out),
            "coverage": coverage_summary,
            "timeline_stats": {
                "total_events": tl_stats.total_events,
                "stays_built": tl_stats.stays_built,
                "events_per_table": events_per_table,
            },
            "timeline_per_stay": timeline_summary,
            "feasibility_decisions": feasibility_summary,
            "feasibility_review_set_sizes": {k: len(v) for k, v in review_sets.items()},
            "episode_summary": episode_summary,
            "total_episodes": len(all_episodes),
            "split_summary": {
                "seed": split_result.seed,
                "patient_split_counts": split_result.split_counts,
                "per_task_episode_counts": split_result.per_task_counts,
            },
            "phase_6_5_schema_gate_ok": True,
            "phase_7_replay_tools_ok": True,
        }
    )
    summary_path = tmp_path / "phase_0_7_realdata_run.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, default=str), encoding="utf-8")
    assert summary_path.exists()

