# CURSOR_EXECUTION_ORDER

Follow this order. Do not skip phases.

1. **Repository skeleton and data access**
   - Create project structure, config, tests, and local MIMIC access.
2. **Task specs**
   - Define versioned specs for hyperkalemia, hypoglycemia, and sustained hypotension.
3. **Normalization codebooks**
   - Build labs/vitals/meds/procedures mappings with ambiguity handling.
4. **Action extraction feasibility checkpoint**
   - Using normalized events, confirm that insulin+dextrose pairings, vasopressor starts/escalations, fluid boluses, and dialysis starts are observable enough to support reliable labels. If not, adjust protocol action families or claims before proceeding.
5. **Canonical timeline generator**
   - Build provenance-preserving, as-of-time-sliceable event timelines.
6. **Deterministic episode generation**
   - Generate trigger windows and separate trigger/observed/protocol labels, keeping negative-window future logic strictly internal to label generation.
7. **Deterministic patient-level splits**
   - Freeze leakage-safe train/val/test manifests and summary stats.
8. **Output schema freezing and runner validation**
   - Freeze the shared JSON output schema for all systems and wire pydantic validation into replay tools, baselines, and agent runners before building them.
9. **Replay environment and structured tools**
   - Expose timestamp-bounded JSON tools only, with provenance and time ranges; never surface negative-window-only future logic.
10. **Baselines**
    - Implement rule, tabular ML (using a deterministic feature-table builder), single-pass LLM, and simple RAG with the shared schema.
11. **Gemini constrained agent**
    - Implement a bounded planner/executor loop plus verifier/reporter pipeline using direct Gemini API calls and pydantic models (no LangChain in v1).
12. **Evaluation**
    - Score trigger quality, action agreement, evidence retrieval, safety, and workflow behavior.
13. **Ablations**
    - Isolate contribution of planner/verifier/tools/notes/abstention and tool-budget limits.
14. **Error analysis**
    - Bucket failures and produce inspectable case-level analyses.
15. **Packaging and reproducibility**
    - Ship reproducible scripts, configs, artifacts, and documentation.

## Execution Gates
- Do not move to a later stage while earlier-stage invariants are failing.
- If labels, leakage control, or event normalization are weak, stop and fix before agent expansion.
