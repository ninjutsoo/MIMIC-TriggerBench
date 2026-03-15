## check-implement

Goal: verify that **all phases from 0 up to the highest `[DONE]` phase in `@IMPLEMENTATION_TASKS.md` work together end-to-end on real MIMIC data**.

### What this command should do
- Read `IMPLEMENTATION_TASKS.md` and note the highest phase that is marked `[DONE]` (call it **N**).
- Use a single dedicated integration test file (see below) to exercise that full path on real data:
  - load settings and auto-discover local MIMIC data,
  - run the inventory command,
  - run normalization coverage on a small slice of real tables,
  - build canonical timelines for a small set of ICU stays,
  - run the Phase 3.5 feasibility checkpoint detectors on those timelines.
- The integration test must be **thorough and inspectable**:
  - log each step start/end with elapsed seconds and key counts (rows scanned, stays built, events built, detections),
  - enforce explicit per-step timeouts so a “stuck” step is handled deterministically,
  - on timeout/hang: dump **all thread stack traces** to a file and fail the test,
  - write a single JSON “run summary” artifact that records the exact inputs used and the full outputs produced.

### Integration test pattern
- For a highest completed phase **N**, keep exactly one permanent real-data integration test named:
  - `tests/test_phase_0_N_realdata.py`
- That test must:
  - use `load_settings()` / repo-local auto-discovery for MIMIC data,
  - keep row counts and number of stays bounded for speed *but still on real MIMIC data*,
  - assert that each step above runs without error and produces non-empty, schema-correct outputs,
  - explicitly capture and assert **inputs** and **outputs** end-to-end (not “smoke-test only”).

### Inspectability artifacts (must exist on non-skip runs)
- `tmp_path/phase_0_N_realdata_run.json`: inputs + outputs + timings
- `tmp_path/phase_0_N_realdata_stuck_stacks.txt`: written on timeout to show where it hung

### How to run and interpret
- From the repo root, for highest completed phase N:
  - `python -m pytest tests/test_phase_0_N_realdata.py -v`

Interpretation:
- **Pass (no skips)**: phases 0–N are wired together correctly on real data.
- **Fail**: investigate the failing step (inventory, normalization, timelines, feasibility, etc.) and fix the implementation until the test passes.
- **Skip**: only acceptable when no usable local MIMIC data is available (no env config and no repo-local mirror). In all other cases, prefer fixing data/path issues so the test can run to completion.
