from pathlib import Path

import os

import pandas as pd
import pytest

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
    assert "schema_ok" in text
    assert "Schema-invalid tables" in text


def test_inventory_report_marks_schema_invalid_tables(tmp_path: Path) -> None:
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir(parents=True)
    pd.DataFrame([{"subject_id": 10, "hadm_id": 1, "admittime": "2020-01-01 00:00:00"}]).to_csv(
        hosp_dir / "admissions.csv.gz",
        index=False,
        compression="gzip",
    )

    settings = Settings(backend="files", mimic_root=tmp_path)
    out = tmp_path / "inventory.md"
    generate_inventory_report(settings, out)

    text = out.read_text(encoding="utf-8")
    assert "`admissions`" in text
    assert "dischtime" in text
    assert "Schema-invalid tables: **1**" in text


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


def test_cli_mapping_ledger_check_passes() -> None:
    code = cli_main(["mapping-ledger-check"])
    assert code == 0


def test_load_table_dataframe_files_backend(tmp_path: Path) -> None:
    # Create a tiny admissions.csv.gz under the expected layout
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir(parents=True)
    csv_path = hosp_dir / "admissions.csv.gz"
    # Write valid gzip-compressed CSV content
    import gzip

    with gzip.open(csv_path, "wb") as f:
        f.write(
            b"subject_id,hadm_id,admittime,dischtime\n"
            b"10,1,2020-01-01 00:00:00,2020-01-02 00:00:00\n"
            b"20,2,2020-01-03 00:00:00,2020-01-04 00:00:00\n"
        )

    settings = Settings(backend="files", mimic_root=tmp_path)
    df = load_table_dataframe(settings, "admissions")
    assert list(df.columns) == ["subject_id", "hadm_id", "admittime", "dischtime"]
    assert len(df) == 2


def test_load_table_dataframe_prefers_parquet_over_csv(tmp_path: Path) -> None:
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir(parents=True)

    pd.DataFrame(
        [{"subject_id": 10, "hadm_id": 1, "admittime": "2020-01-01 00:00:00", "dischtime": "2020-01-02 00:00:00"}]
    ).to_parquet(hosp_dir / "admissions.parquet", index=False)

    pd.DataFrame(
        [
            {"subject_id": 10, "hadm_id": 1, "admittime": "2020-01-01 00:00:00", "dischtime": "2020-01-02 00:00:00"},
            {"subject_id": 20, "hadm_id": 2, "admittime": "2020-01-03 00:00:00", "dischtime": "2020-01-04 00:00:00"},
        ]
    ).to_csv(hosp_dir / "admissions.csv.gz", index=False, compression="gzip")

    settings = Settings(backend="files", mimic_root=tmp_path)
    df = load_table_dataframe(settings, "admissions")
    assert len(df) == 1


def test_load_table_dataframe_csv_fallback(tmp_path: Path) -> None:
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"subject_id": 10, "hadm_id": 1, "admittime": "2020-01-01 00:00:00", "dischtime": "2020-01-02 00:00:00"}]
    ).to_csv(hosp_dir / "admissions.csv", index=False)

    settings = Settings(backend="files", mimic_root=tmp_path)
    df = load_table_dataframe(settings, "admissions")
    assert len(df) == 1
    assert set(df.columns) == {"subject_id", "hadm_id", "admittime", "dischtime"}


def test_load_table_dataframe_schema_validation_failure(tmp_path: Path) -> None:
    hosp_dir = tmp_path / "hosp"
    hosp_dir.mkdir(parents=True)
    pd.DataFrame([{"subject_id": 10, "hadm_id": 1, "admittime": "2020-01-01 00:00:00"}]).to_csv(
        hosp_dir / "admissions.csv.gz",
        index=False,
        compression="gzip",
    )
    settings = Settings(backend="files", mimic_root=tmp_path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_table_dataframe(settings, "admissions")
