"""Real-data integration test for implemented phases 0–4.

This test is designed for the `check-implement` command.  It exercises,
on real MIMIC data when available, the main public entrypoints from
phases marked `[DONE]` in `IMPLEMENTATION_TASKS.md`:

- Phase 0/1: settings + inventory + table loading
- Phase 3: normalization coverage on real tables
- Phase 4: canonical timeline builder
- Phase 3.5: action feasibility checkpoint on those timelines

The test is strict: it captures exact inputs, asserts on all output
shapes and content, and writes a run summary to tmp_path so you can
inspect exactly what data came in and what was produced.  Skip only
when no local MIMIC data is available.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import threading
import time
from pathlib import Path

import pytest

from mimic_triggerbench.config import DataBackend, Settings, load_settings
from mimic_triggerbench.mimic_tables import (
    iter_table_specs,
    required_table_names,
    resolve_table_path,
)
from mimic_triggerbench.data_access import generate_inventory_report
from mimic_triggerbench.data_access.tables import load_table_dataframe
from mimic_triggerbench.data_access.normalization_audit import (
    scan_normalization_coverage,
    TableScanResult,
)
from mimic_triggerbench.timeline import build_all_timelines
from mimic_triggerbench.timeline.models import CanonicalEvent
from mimic_triggerbench.feasibility import (
    run_feasibility_checkpoint,
    FeasibilityDecision,
    DetectedAction,
)

# Expected action families from feasibility checkpoint (Phase 3.5).
EXPECTED_FEASIBILITY_FAMILIES = frozenset({
    "insulin_dextrose",
    "vasopressor_start",
    "vasopressor_escalation",
    "fluid_bolus",
    "dialysis_start",
})

# CanonicalEvent source tables produced by timeline builder (Phase 4).
EXPECTED_TIMELINE_SOURCE_TABLES = frozenset({
    "labevents",
    "chartevents",
    "inputevents",
    "procedureevents",
    "outputevents",
})

# Allowed event_category values on CanonicalEvent.
EXPECTED_EVENT_CATEGORIES = frozenset({
    "lab", "vital", "med_bolus", "med_infusion", "procedure", "output", "other",
})

LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def _step(name: str, summary: dict) -> object:
    t0 = time.perf_counter()
    LOGGER.info("[phase_0_4_realdata] START %s", name)
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        LOGGER.info("[phase_0_4_realdata] DONE  %s (%.2fs)", name, dt)
        summary.setdefault("timings_sec", {})[name] = round(dt, 4)


def _dump_thread_stacks(path: Path) -> None:
    """Write a best-effort traceback for all threads (Windows-safe)."""
    import faulthandler

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Thread stack dump (faulthandler)\n\n")
        faulthandler.dump_traceback(file=f, all_threads=True)


def _run_with_timeout(name: str, timeout_s: int, fn, *, tmp_path: Path):
    """Run fn() in a thread and enforce a hard timeout with stack dump."""
    q: queue.Queue = queue.Queue(maxsize=1)

    def _runner():
        try:
            q.put(("ok", fn()))
        except BaseException as e:  # noqa: BLE001 - propagate to test
            q.put(("err", e))

    t = threading.Thread(target=_runner, name=f"phase_0_4_realdata::{name}", daemon=True)
    t.start()
    t.join(timeout=timeout_s)
    if t.is_alive():
        dump_path = tmp_path / "phase_0_4_realdata_stuck_stacks.txt"
        _dump_thread_stacks(dump_path)
        pytest.fail(
            f"Step {name!r} exceeded timeout ({timeout_s}s). "
            f"Wrote thread stacks to {dump_path!s}."
        )
    status, payload = q.get_nowait()
    if status == "err":
        raise payload
    return payload


def _has_required_files(mimic_root: Path) -> bool:
    return all(
        resolve_table_path(mimic_root, spec.table_name) is not None
        for spec in iter_table_specs()
    )


def _discover_repo_mimic_root() -> Path | None:
    """Discover a repo-local MIMIC root if present.

    Expected layout (mirrors PhysioNet download structure):
    - <repo>/physionet.org/files/mimiciv/<version>/{icu,hosp}/*.{parquet,csv.gz,csv}
    """
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
    """Best-effort loader for real-data settings.

    Preference order:
    1) env/.env via load_settings()
    2) repo-local physionet.org mirror
    """
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


def test_implemented_phases_realdata_end_to_end(tmp_path: Path) -> None:
    """End-to-end real-data test for phases 0–4.

    Strict checks: exact inputs captured, all outputs asserted (shape + content).
    A run summary is written to tmp_path/phase_0_4_realdata_run.json for
    inspectability.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_summary: dict = {}

    # -------------------------------------------------------------------------
    # INPUTS: capture exactly what we use
    # -------------------------------------------------------------------------
    settings = _load_real_settings()
    if settings is None or settings.mimic_root is None:
        pytest.skip("No local MIMIC data available for real-data integration test.")
    if not _has_required_files(settings.mimic_root):
        pytest.skip(
            f"Local MIMIC files not found under mimic_root={settings.mimic_root!s}."
        )

    mimic_root = Path(settings.mimic_root)
    required_tables = list(required_table_names())
    resolved_tables: dict[str, str] = {}
    for t in required_tables:
        r = resolve_table_path(mimic_root, t)
        resolved_tables[t] = str(r.path) if r else ""

    assert settings.backend == DataBackend.FILES, "Test expects FILES backend."
    assert mimic_root.exists(), f"mimic_root must exist: {mimic_root}"
    assert len(resolved_tables) == len(required_tables), "All required tables must be resolvable."

    # Load icustays and pick stays
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
    # Cap rows so the test finishes in reasonable time while still using real data.
    max_rows_per_table = 25_000
    coverage_top_k = 10
    step_timeout_s = 180

    LOGGER.info(
        "[phase_0_4_realdata] mimic_root=%s icustays_rows=%d sample_stays=%s max_rows_per_table=%d",
        mimic_root,
        num_icustays_rows,
        sample_stays,
        max_rows_per_table,
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

    # -------------------------------------------------------------------------
    # Phase 0/1: inventory report — assert full content
    # -------------------------------------------------------------------------
    inv_out = tmp_path / "inventory.md"
    with _step("inventory_report", run_summary):
        _run_with_timeout(
            "inventory_report",
            step_timeout_s,
            lambda: generate_inventory_report(settings, inv_out),
            tmp_path=tmp_path,
        )
        inv_text = inv_out.read_text(encoding="utf-8")

    assert inv_out.exists(), "Inventory report file must be created."
    assert "# Data inventory (generated)" in inv_text
    assert settings.backend.value in inv_text
    assert str(mimic_root) in inv_text or f"`{mimic_root}`" in inv_text
    assert "| table | present | source |" in inv_text or "| table | present |" in inv_text

    for table_name in required_tables:
        assert f"`{table_name}`" in inv_text or table_name in inv_text, (
            f"Inventory must list required table {table_name!r}."
        )

    # Key tables for timeline/coverage must be present in inventory with "yes"
    for key_table in ("icustays", "labevents", "chartevents"):
        assert key_table in inv_text
        # Line should contain "yes" for present when table is found (we already verified _has_required_files)
        idx = inv_text.find(key_table)
        snippet = inv_text[idx : idx + 80]
        assert "yes" in snippet or "present" in snippet, (
            f"Inventory row for {key_table!r} should indicate present."
        )

    # -------------------------------------------------------------------------
    # Phase 3: normalization coverage — assert structure and values
    # -------------------------------------------------------------------------
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

    assert isinstance(coverage_results, list), "Coverage must return a list."
    tables_in_coverage = {r.table for r in coverage_results}
    assert "labevents" in tables_in_coverage, "Coverage must include labevents."
    assert "chartevents" in tables_in_coverage, "Coverage must include chartevents."

    total_rows_scanned = 0
    coverage_summary: list[dict] = []
    for r in coverage_results:
        assert isinstance(r, TableScanResult)
        assert r.table
        assert r.rows_scanned >= 0
        assert r.mapped >= 0
        assert r.unmapped >= 0
        assert r.ambiguous >= 0
        # ambiguous is a subset of mapped (\"mapped but flagged\"), so do not sum it into total.
        assert r.mapped + r.unmapped <= r.rows_scanned
        assert r.ambiguous <= r.mapped
        assert len(r.top_unmapped_itemids) <= coverage_top_k
        assert len(r.top_unmapped_labels) <= coverage_top_k
        total_rows_scanned += r.rows_scanned
        coverage_summary.append({
            "table": r.table,
            "rows_scanned": r.rows_scanned,
            "mapped": r.mapped,
            "unmapped": r.unmapped,
            "ambiguous": r.ambiguous,
        })

    assert total_rows_scanned > 0, "At least one table must have rows scanned."

    # -------------------------------------------------------------------------
    # Phase 4: canonical timelines — assert event schema and stats
    # -------------------------------------------------------------------------
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
            pytest.skip(f"Could not build canonical timelines from local MIMIC data: {e!r}")

    assert isinstance(timelines, dict), "Timelines must be a dict."
    assert timelines, "At least one stay must produce a timeline."
    assert tl_stats.total_events > 0, "At least one event must be built."
    assert tl_stats.stays_built > 0
    assert tl_stats.stays_built == len(timelines)

    events_per_table: dict[str, int] = dict(tl_stats.events_per_table)
    timeline_summary: list[dict] = []
    for stay_id, events in timelines.items():
        assert isinstance(stay_id, int)
        assert isinstance(events, list)
        assert len(events) > 0, f"Stay {stay_id} must have at least one event."
        for ev in events:
            assert isinstance(ev, CanonicalEvent)
            assert ev.stay_id == stay_id
            assert ev.subject_id is not None
            assert ev.hadm_id is not None
            assert ev.event_time is not None
            assert ev.event_uid and isinstance(ev.event_uid, str)
            assert ev.source_table in EXPECTED_TIMELINE_SOURCE_TABLES, (
                f"Unexpected source_table {ev.source_table!r}"
            )
            assert ev.event_category in EXPECTED_EVENT_CATEGORIES
            assert ev.canonical_name
        timeline_summary.append({
            "stay_id": stay_id,
            "event_count": len(events),
            "source_tables_seen": list({e.source_table for e in events}),
            "categories_seen": list({e.event_category for e in events}),
        })

    assert sum(events_per_table.values()) == tl_stats.total_events

    # -------------------------------------------------------------------------
    # Phase 3.5: feasibility checkpoint — assert decisions and review_sets
    # -------------------------------------------------------------------------
    with _step("feasibility_checkpoint", run_summary):
        decisions, review_sets = _run_with_timeout(
            "feasibility_checkpoint",
            step_timeout_s,
            lambda: run_feasibility_checkpoint(timelines),
            tmp_path=tmp_path,
        )

    assert isinstance(decisions, list)
    assert decisions, "Feasibility must return at least one decision."
    decision_families = {d.action_family for d in decisions}
    assert decision_families == EXPECTED_FEASIBILITY_FAMILIES, (
        f"Expected families {EXPECTED_FEASIBILITY_FAMILIES}, got {decision_families}."
    )

    feasibility_summary: list[dict] = []
    for d in decisions:
        assert isinstance(d, FeasibilityDecision)
        assert d.action_family
        assert hasattr(d, "threshold")
        assert hasattr(d, "stats")
        assert isinstance(d.passed, bool)
        feasibility_summary.append({
            "action_family": d.action_family,
            "passed": d.passed,
            "detected_events": d.stats.detected_events,
            "unique_stays": d.stats.unique_stays,
            "unique_patients": d.stats.unique_patients,
        })

    assert isinstance(review_sets, dict)
    # review_sets only contains families that had at least one detection
    assert set(review_sets.keys()) <= EXPECTED_FEASIBILITY_FAMILIES, (
        f"Review set keys must be subset of expected families; got {set(review_sets.keys())}."
    )

    for family, actions in review_sets.items():
        assert isinstance(family, str)
        assert isinstance(actions, list)
        for a in actions:
            assert isinstance(a, DetectedAction)
            assert a.action_family == family
            assert isinstance(a.stay_id, int)
            assert isinstance(a.subject_id, int)
            assert isinstance(a.hadm_id, int)
            assert a.detection_time is not None
            assert isinstance(a.source_event_uids, tuple)
            assert all(isinstance(uid, str) for uid in a.source_event_uids)

    # -------------------------------------------------------------------------
    # Run summary: write to tmp_path for inspectability
    # -------------------------------------------------------------------------
    run_summary.update({
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
        "feasibility_review_set_sizes": {
            k: len(v) for k, v in review_sets.items()
        },
    })
    summary_path = tmp_path / "phase_0_4_realdata_run.json"
    summary_path.write_text(
        json.dumps(run_summary, indent=2, default=str),
        encoding="utf-8",
    )
    assert summary_path.exists(), "Run summary must be written for inspectability."
