# PROJECT_GROUND_TRUTH

## Project Summary
MIMIC-TriggerBench is an event-triggered ICU monitoring and escalation benchmark plus a constrained agent on MIMIC-IV. The benchmark is the primary contribution; the agent is evaluated within that benchmark contract.

## v1 Scope (Locked)
v1 includes exactly three task families:
1. moderate-to-severe hyperkalemia response (K >= 6.0 mmol/L, clinically important range)
2. severe (Level 2) hypoglycemia response (glucose < 54 mg/dL)
3. sustained hypotension response (MAP < 65 mmHg for at least 15 minutes)

The project must not drift into a generic ICU copilot in v1.

## Non-Negotiable Invariants
- No future leakage.
- Deterministic labels and reproducible splits.
- No clinician-style free-text judgment as the main label.
- Protocol-derived gold action labels are separate from observed clinician actions.
- Replay/tool interface is structured, timestamp-bounded, and auditable.
- Benchmark outputs are strict machine-scorable JSON.
- No vague free-text benchmark outputs.

## Safe Paper Claims
- Reproducible event-triggered ICU benchmark on MIMIC-IV.
- Evaluation includes workflow behavior (evidence retrieval, tool use, abstention, action grounding), not only final prediction quality.
- Constrained tool-using systems can be compared fairly against non-agent baselines under one shared contract.

## Forbidden Claim Patterns
- Claims of physician-level equivalence.
- Claims of bedside deployment readiness.
- Broad "general clinical reasoning" claims outside benchmark tasks.
- Claims unsupported by benchmark artifacts, metrics, or ablations.

## Required Baselines
- Rule baseline
- Tabular ML baseline
- Single-pass LLM baseline
- Simple RAG baseline

Baselines must run before expanding agent complexity.

## Agent Architecture Contract
Recommended constrained 4-module pipeline:
- planner
- retriever/executor
- verifier
- reporter

This pipeline must remain bounded, structured, and benchmark-driven.

## Output Schema Contract
All systems (baselines and agent) must emit the same strict final JSON schema. No free-form benchmark outputs are allowed as primary outputs.

## Notes Policy
Start with structured data only. Notes are off by default and may be introduced later as an explicit ablation/extension after structured tasks are stable.

## Reproducibility Expectations
- Version task specs, mappings, and split manifests.
- Keep deterministic generation paths and fixed seeds where applicable.
- Preserve provenance and time-range metadata in replay/tool outputs.
- Maintain tests for leakage prevention, schema stability, and split safety.
- Update docs and audit artifacts whenever episode semantics change.
