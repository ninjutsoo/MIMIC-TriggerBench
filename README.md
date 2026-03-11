## MIMIC-TriggerBench

Benchmark-first project for event-triggered ICU escalation agents on MIMIC-IV.

### Environment

- **Python**: 3.11
- **Packaging**: `pyproject.toml` (install with `pip install -e .[dev]` or `uv pip install -e .[dev]`)

Recommended virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -e .[dev]
```

### Layout (early skeleton)

- `src/mimic_triggerbench/`
  - `config/` – config models and loaders
  - `data_access/` – raw MIMIC access, local/SQL backends
  - `labeling/` – task specs and deterministic label generation
  - `replay/` – canonical timelines and replay environment
  - `baselines/` – rule/ML/LLM baselines
  - `agent/` – Gemini-based tool-using agent
  - `evaluation/` – metrics and reports
- `docs/` – project docs and inventories
- `scripts/` – helper scripts (e.g. dataset download)
- `tests/` – basic smoke tests and placeholders

This README will be expanded once the core benchmark artifacts exist (task specs, codebooks, replay environment, and episodes).

