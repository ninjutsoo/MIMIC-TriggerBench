"""Integration test: normalization audit runs on real local MIMIC data if present.

Policy:
- If MIMIC data is available (either via env settings OR via the repo-local
  `physionet.org/files/mimiciv/<version>/` layout), this test must run.
- If no local data is available, the test may skip (to keep CI environments
  without licensed data green).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mimic_triggerbench.config import DataBackend, Settings, load_settings
from mimic_triggerbench.mimic_tables import iter_table_specs, resolve_table_path
from mimic_triggerbench.data_access.normalization_audit import (
    scan_normalization_coverage,
    write_normalization_coverage_report,
)


def _has_required_files(mimic_root: Path) -> bool:
    return all(resolve_table_path(mimic_root, spec.table_name) is not None for spec in iter_table_specs())


def _discover_repo_mimic_root() -> Path | None:
    """Discover a repo-local MIMIC root if present.

    Expected local layout (mirrors PhysioNet download structure):
    - <repo>/physionet.org/files/mimiciv/<version>/{icu,hosp}/*.{parquet,csv.gz,csv}
    """
    repo_root = Path(__file__).resolve().parents[1]
    base = repo_root / "physionet.org" / "files" / "mimiciv"
    if not base.exists():
        return None

    # Prefer the newest version directory if multiple are present.
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None

    for version_dir in sorted(candidates, key=lambda p: p.name, reverse=True):
        if _has_required_files(version_dir):
            return version_dir
    return None


def test_normalization_audit_runs_on_local_data(tmp_path: Path) -> None:
    settings: Settings | None = None

    # 1) Prefer explicit settings from env vars when configured.
    try:
        settings = load_settings(dotenv_path=None)
    except Exception:
        settings = None

    # 2) Otherwise auto-discover repo-local data (common for this workspace).
    if settings is None or settings.mimic_root is None or not _has_required_files(settings.mimic_root):
        discovered = _discover_repo_mimic_root()
        if discovered is not None:
            settings = Settings(backend=DataBackend.FILES, mimic_root=discovered)

    if settings is None or settings.mimic_root is None:
        pytest.skip("No local MIMIC data found via env vars or repo-local physionet.org mirror.")
    if not _has_required_files(settings.mimic_root):
        pytest.skip(f"Local MIMIC files not found under mimic_root={settings.mimic_root!s}.")

    results = scan_normalization_coverage(settings, max_rows_per_table=50_000, top_k=10)
    assert len(results) >= 2
    assert any(r.rows_scanned > 0 for r in results)
    # Ensure the scan is actually surfacing unmapped high-frequency terms.
    assert any(len(r.top_unmapped_itemids) > 0 or len(r.top_unmapped_labels) > 0 for r in results)

    out = tmp_path / "normalization_coverage.md"
    write_normalization_coverage_report(
        results,
        out,
        settings=settings,
        max_rows_per_table=50_000,
        top_k=10,
    )
    text = out.read_text(encoding="utf-8")
    assert "Normalization coverage" in text
    assert "Top unmapped" in text
