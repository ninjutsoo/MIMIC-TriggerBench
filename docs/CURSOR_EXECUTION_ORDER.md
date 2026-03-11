# CURSOR_EXECUTION_ORDER

Follow this order. Do not skip phases.

1. **Repository skeleton and data access**
   - Create project structure, config, tests, and local MIMIC access.
2. **Task specs**
   - Define versioned specs for hyperkalemia, hypoglycemia, and sustained hypotension.
3. **Normalization codebooks**
   - Build labs/vitals/meds/procedures mappings with ambiguity handling.
4. **Canonical timeline generator**
   - Build provenance-preserving, as-of-time-sliceable event timelines.
5. **Deterministic episode generation**
   - Generate trigger windows and separate trigger/observed/protocol labels.
6. **Deterministic patient-level splits**
   - Freeze leakage-safe train/val/test manifests and summary stats.
7. **Replay environment and structured tools**
   - Expose timestamp-bounded JSON tools only, with provenance and time ranges.
8. **Baselines**
   - Implement rule, tabular ML, single-pass LLM, and simple RAG with shared schema.
9. **Gemini constrained agent**
   - Implement bounded planner -> retriever/executor -> verifier -> reporter pipeline.
10. **Evaluation**
    - Score trigger quality, action agreement, evidence retrieval, safety, and workflow behavior.
11. **Ablations**
    - Isolate contribution of planner/verifier/tools/notes/abstention and tool-budget limits.
12. **Error analysis**
    - Bucket failures and produce inspectable case-level analyses.
13. **Packaging and reproducibility**
    - Ship reproducible scripts, configs, artifacts, and documentation.

## Execution Gates
- Do not move to a later stage while earlier-stage invariants are failing.
- If labels, leakage control, or event normalization are weak, stop and fix before agent expansion.
