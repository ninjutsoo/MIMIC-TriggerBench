# Data inventory (manual notes)

This is the authoritative manual note for the local dataset setup used by this repo.

## Local setup

- **MIMIC-IV version**: `mimiciv/3.1`
- **MIMIC-IV-Note version**: `mimic-iv-note/2.2`
- **Source**: local files (Option B: DuckDB used as embedded query engine when needed)
- **MIMIC-IV path**: `C:\Users\mrosh\OneDrive\Documents\GitHub\MIMIC-TriggerBench\physionet.org\files\mimiciv\3.1`
- **MIMIC-IV-Note path**: `C:\Users\mrosh\OneDrive\Documents\GitHub\MIMIC-TriggerBench\physionet.org\files\mimic-iv-note\2.2`
- **Postgres**: not used currently

## Which raw tables are available?

Record any missing tables or naming differences here (this is the authoritative manual note).

- ICU module missing: none (per `docs/data_inventory_generated.md`)
- Hosp module missing: none (per `docs/data_inventory_generated.md`)
- Notes available: yes (MIMIC-IV-Note 2.2 present on disk; exact note table coverage to be audited when note support is introduced)

## Notes

- No schema names (file backend).
- Raw data is `*.csv.gz` as downloaded from PhysioNet.

