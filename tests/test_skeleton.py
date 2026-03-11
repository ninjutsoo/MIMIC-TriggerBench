from pathlib import Path

import os

from mimic_triggerbench.config import Settings, load_settings
from mimic_triggerbench.data_access import generate_inventory_report, load_table_dataframe
from mimic_triggerbench.cli import main as cli_main


def test_inventory_report_runs(tmp_path: Path) -> None:
    settings = Settings(backend="files", mimic_root=tmp_path)
    out = tmp_path / "inventory.md"
    generate_inventory_report(settings, out)
    # File should be created even if no tables are present
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "Data inventory" in text


def test_load_settings_from_env_files_backend(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MIMIC_BACKEND", "files")
    monkeypatch.setenv("MIMIC_ROOT", str(tmp_path))
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    s = load_settings(dotenv_path=None)
    assert s.backend.value == "files"
    assert s.mimic_root == tmp_path


def test_cli_inventory_writes_report(tmp_path: Path, monkeypatch) -> None:
    # Avoid reading real .env; use env vars and write report to tmp
    monkeypatch.setenv("MIMIC_BACKEND", "files")
    monkeypatch.setenv("MIMIC_ROOT", str(tmp_path))
    out = tmp_path / "docs" / "data_inventory_generated.md"
    code = cli_main(["inventory", "--dotenv", "does-not-exist.env", "--out", str(out)])
    assert code == 0
    assert out.exists()


def test_load_table_dataframe_files_backend(tmp_path: Path) -> None:
    # Create a tiny admissions.csv.gz under the expected layout
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir(parents=True)
    csv_path = hosp_dir / "admissions.csv.gz"
    # Write valid gzip-compressed CSV content
    import gzip

    with gzip.open(csv_path, "wb") as f:
        f.write(b"hadm_id,subject_id\n1,10\n2,20\n")

    settings = Settings(backend="files", mimic_root=tmp_path)
    df = load_table_dataframe(settings, "admissions")
    assert list(df.columns) == ["hadm_id", "subject_id"]
    assert len(df) == 2

