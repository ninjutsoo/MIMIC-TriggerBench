# IMPLEMENTATION_TASKS.md

## Project
MIMIC-TriggerBench: Event-Triggered ICU Monitoring and Escalation Agent on MIMIC-IV

## Goal
Build a reproducible benchmark and agentic system that replays ICU stays from MIMIC-IV as a time-ordered stream. When a clinically defined trigger fires, the system must gather evidence, run tools/calculators, check protocol constraints, and output a structured escalation recommendation with traceable evidence.

This document separates:
- **FOR ME**: tasks the researcher should do manually
- **FOR CURSOR**: exact implementation tasks to give Cursor
- **DONE CRITERIA**: what must be true before moving on

Do not skip phases. The project will fail if labels, leakage control, or event normalization are weak.

---

# Phase 0. Scope lock and project rules

## Objective
Freeze a narrow, paper-worthy scope before coding expands into a generic ICU copilot.

## Scope for v1 paper
Implement exactly these three task families first:
1. Severe hyperkalemia response
2. Severe hypoglycemia response
3. Sustained hypotension response

Optional later extension only after v1 is stable:
4. Sepsis-style screening or deterioration workflow

## Non-negotiable rules
- No future leakage
- No clinician-style free-text judgment as the main label
- All labels must be deterministic and reproducible
- All agent outputs must follow a strict JSON schema
- Start with structured data first, then add note support as an ablation or extension
- Benchmark first, agent second

## FOR ME
- Create a new repository called `mimic-triggerbench`
- Decide whether the first submission target is CHIL/MLHC/JAMIA or a benchmark-style track
- Freeze trigger definitions in a separate versioned document before training anything
- Decide local environment: Linux/macOS/WSL, Python version, package manager

## FOR CURSOR
**Prompt to Cursor:**
Create the initial repository structure for a benchmark-first medical agent project called `mimic-triggerbench`. Use Python 3.11, uv or conda-compatible packaging, pytest, pydantic, pandas, duckdb, sqlite support, pyarrow, scikit-learn, xgboost, and a clean modular structure. Add a README with setup instructions and placeholder modules for data extraction, labeling, replay environment, baselines, agents, and evaluation.

## DONE CRITERIA
- Repo created
- Folder structure exists
- Tests run successfully with placeholders
- README explains local setup

---

# Phase 1. Local environment and dataset access

## Objective
Make the offline MIMIC-IV and MIMIC-IV-Note data queryable on your machine.

## Recommended local setup
Use one of these:
- **Option A**: PostgreSQL if you already loaded MIMIC there
- **Option B**: DuckDB + parquet/csv extracts if data is local and easier to manage
- **Option C**: SQLite only for small derived benchmark tables, not full raw event data

For fastest research iteration, a good pattern is:
- Raw source in local CSV/Parquet or PostgreSQL
- Intermediate normalized tables in DuckDB/Parquet
- Final benchmark episodes in Parquet/JSONL

## Required raw datasets
You need these local tables from MIMIC-IV:

### ICU module
- `icustays`
- `chartevents`
- `inputevents`
- `outputevents`
- `procedureevents`

### Hospital module
- `admissions`
- `patients`
- `labevents`
- `prescriptions`
- `emar` if available in your local version
- `pharmacy` if available in your local version
- `transfers`
- `diagnoses_icd` optional for analysis

### MIMIC-IV-Note optional for phase 1b or later
- `discharge`
- radiology/nursing/other note tables if available in your release

## FOR ME
1. Put the dataset on disk under a stable path, for example:
   ```
   /data/mimiciv/
   /data/mimiciv_note/
   ```
2. Confirm table names and file names in your local copy.
3. Create a `.env` file with:
   ```
   MIMIC_ROOT=/data/mimiciv
   MIMIC_NOTE_ROOT=/data/mimiciv_note
   GEMINI_API_KEY=...
   ```
4. Choose whether raw queries will hit PostgreSQL or local files.
5. Write down your actual MIMIC version and any missing tables in `docs/data_inventory.md`.

## FOR CURSOR
**Prompt to Cursor:**
Build a data access layer for local offline MIMIC-IV and optional MIMIC-IV-Note data. It must support either local CSV/Parquet files or PostgreSQL via a config flag. Add a config-driven loader, schema validation, and a command that prints which required tables are found and which are missing. Output a markdown inventory report under `docs/data_inventory_generated.md`.

## DONE CRITERIA
- You can programmatically load all required tables
- Data inventory report exists
- Missing tables are explicitly logged
- No agent code yet

---

# Phase 2. Clinical task specification and trigger definitions

## Objective
Define clinically defensible triggers and action windows before label generation.

## Task 1. Moderate-to-severe hyperkalemia (clinically important)

### Proposed trigger
At time `t`, trigger when the latest potassium value up to `t` is `>= 6.0 mmol/L` (moderate-to-severe hyperkalemia per UK Kidney Association guidance, with severe commonly defined as `>= 6.5 mmol/L`).

### Exclusion rules
- Chronic dialysis / ESRD on maintenance RRT prior to admission (the benchmark’s action labels are meant to reflect acute escalation, not routine maintenance dialysis care)
- Comfort measures only (CMO) / end-of-life care flags if available
- Hemolyzed specimen if explicitly flagged in the lab record (pseudo-hyperkalemia)

### Supporting context to retrieve
- Most recent potassium values in past 24h
- Most recent creatinine/BUN in past 24h
- Recent insulin, dextrose, calcium, bicarbonate, potassium-binding drugs in past 6h
- Dialysis/procedure indicators in past 12h
- Recent urine output if used for severity context

### Acceptable downstream actions within window
Window: 0 to 6 hours after trigger
- calcium administration (e.g., IV calcium gluconate > 1g or IV calcium chloride > 0.5g)
- insulin + dextrose (e.g., IV regular/rapid-acting insulin + IV dextrose >= 25g)
- bicarbonate IV
- potassium-lowering therapy (e.g., nebulized albuterol; potassium binders if present in your local tables)
- dialysis-related action (e.g., new RRT initiation / dialysis procedure event)
- repeat potassium measurement can be a secondary action but should not be the only “treatment” label

## Task 2. Severe hypoglycemia (ADA Level 2)

### Proposed trigger
At time `t`, trigger when the latest glucose value up to `t` is `< 54 mg/dL` (ADA Standards of Care Level 2 hypoglycemia, “clinically important hypoglycemia”).

### Exclusion rules
- Comfort measures only (CMO) / end-of-life care flags if available

### Supporting context to retrieve
- Most recent glucose trend in past 6h
- Insulin administration in past 6h
- Dextrose/glucagon in past 2h
- Nutrition or dextrose infusions if available

### Acceptable downstream actions within window
Window: 0 to 1 hour after trigger
- IV dextrose (>= 10g; allow D50 amp or D10 bolus depending on local practice/recording)
- glucagon (IM/IV 1mg)
- correction documented via medication/administration record
- repeat glucose within short window can be secondary evidence

## Task 3. Sustained hypotension

### Proposed trigger
At time `t`, trigger when MAP `< 65` sustained for at least 15 minutes using charted events available up to `t` (aligned with anesthesia quality metrics and outcome literature showing increased organ-injury risk after ~13–15 minutes below 65 mmHg).

### Exclusion rules
- Comfort measures only (CMO) / end-of-life care flags if available
- Mechanical circulatory support with non-pulsatile flow (e.g., LVAD/Impella), if identifiable

### Episode clustering rule (deterministic)
- After a hypotension trigger fires, do not fire another episode for the same stay until there has been at least 60 minutes of MAP >= 65 (“washout” period).

### Supporting context to retrieve
- MAP trend past 60 min
- Vasopressors past 2h
- IV fluids/boluses past 2h
- Lactate if available past 6h
- Urine output past 6h optional

### Acceptable downstream actions within window
Window: 0 to 1 hour after trigger
- fluid bolus (>= 250 mL crystalloid administered within <= 30 minutes, or explicitly labeled as a bolus)
- vasopressor start
- vasopressor escalation (e.g., dose increase > 20% within the window; define relative to your available dose/rate fields)
- repeat hemodynamic monitoring may be secondary, not sufficient alone

## Negative window definition (Phase 2)
Define “negative episodes” as decision times `t` where the patient is “at risk” (ICU stay ongoing and the signal/lab is measured) but:
- Trigger is false at `t`
- Trigger has been false for the prior 2 hours
- Trigger remains false for the subsequent 6 hours (avoid “imminent trigger” near-miss leakage)

## FOR ME
- Review these definitions manually and decide exact thresholds/window values
- Keep a changelog in `docs/trigger_definitions.md`
- Freeze version `v0.1` once approved
- If you want stronger clinical credibility, have one clinician or medically knowledgeable collaborator comment on the definitions and save that feedback

## FOR CURSOR
**Prompt to Cursor:**
Create versioned task-spec YAML files for hyperkalemia, hypoglycemia, and sustained hypotension. Each spec must include trigger definition, supporting evidence types, exclusion rules, accepted action families, action windows, and negative window definitions. Add pydantic validation and tests that reject malformed task specs.

## DONE CRITERIA
- Machine-readable task specs exist
- Definitions are versioned
- Thresholds and windows are frozen for v0.1

---

# Phase 3. Event normalization and codebooks

## Objective
Map raw MIMIC item names and medication names into normalized concepts used for triggers and actions.

## What must be normalized

### Labs
- potassium
- glucose
- creatinine
- BUN
- lactate optional

### Vitals / hemodynamics
- MAP
- systolic BP optional
- heart rate optional

### Medications/interventions
- calcium formulations
- insulin
- dextrose
- glucagon
- bicarbonate
- vasopressors by class
- IV fluid bolus
- dialysis/procedural interventions

## Output format
Create codebooks under:
- `data_access/mappings/labs.yaml`
- `data_access/mappings/vitals.yaml`
- `data_access/mappings/meds.yaml`
- `data_access/mappings/procedures.yaml`

Each mapping should include:
- canonical concept name
- raw labels/itemids matched
- units if relevant
- conversion rule if relevant
- notes on ambiguity

## FOR ME
- Inspect a sample of raw MIMIC rows for each target concept
- Manually verify ambiguous medication names and ICU item mappings
- Log unresolved ambiguities instead of silently forcing mappings

## FOR CURSOR
**Prompt to Cursor:**
Build normalization pipelines and codebooks for labs, hemodynamics, medications, fluids, and procedures required by the three benchmark tasks. The system must map raw MIMIC labels/itemids/drug names into canonical concepts, attach units, flag ambiguous rows, and include tests for mapping coverage and unexpected unmapped high-frequency terms.

## DONE CRITERIA
- Canonical codebooks exist
- High-frequency rows are mapped or explicitly flagged
- Unit normalization is tested
- Ambiguous mappings are tracked

---

# Phase 4. Canonical timeline construction

## Objective
Create a single time-ordered timeline representation per ICU stay.

## Required canonical schema
Each event row should minimally contain:
- `subject_id`
- `hadm_id`
- `stay_id`
- `event_time`
- `source_table`
- `event_category`
- `canonical_name`
- `value_num`
- `value_text`
- `unit`
- `raw_id`
- `raw_label`
- `metadata_json`

## Additional requirements
- Preserve source provenance
- Preserve exact timestamps
- Support multiple events with same timestamp
- Support filtering by event category and canonical name
- Support “as-of time” slicing for replay

## FOR ME
- Decide output storage format for canonical timelines, ideally Parquet
- Confirm enough disk space for derived tables

## FOR CURSOR
**Prompt to Cursor:**
Implement a canonical timeline builder that merges normalized events from required MIMIC tables into a single time-ordered representation per ICU stay. Save outputs as partitioned Parquet by task-relevant cohorts. Include indexing utilities for fast as-of time slicing and unit tests for temporal ordering and provenance preservation.

## DONE CRITERIA
- Canonical timelines exist for sample stays
- As-of slicing works
- Tests verify strict ordering and provenance

---

# Phase 5. Deterministic label generation

## Objective
Generate benchmark episodes and labels with no future leakage.

## Episode unit
A benchmark episode is a decision point where the agent must act.

## Required episode fields
- `episode_id`
- `task_name`
- `subject_id`
- `hadm_id`
- `stay_id`
- `decision_time`
- `context_start`
- `trigger_label`
- `trigger_type`
- `accepted_action_families`
- `observed_action_families`
- `mandatory_evidence_types`
- `negative_or_positive_window`
- `split`

## Label design

### Positive episodes
Trigger condition becomes true at `decision_time`.

### Negative episodes
Matched windows where trigger condition is not met and no qualifying downstream action should be recommended.

### Outcome labels
At minimum generate:
- trigger present or absent
- observed downstream clinician action within task window
- recommended rule-derived action set

### Important
Separate:
1. whether the trigger fired
2. what actions clinicians actually did
3. what the protocol-derived correct action family is

Do not conflate observed clinician behavior with gold protocol.

## FOR ME
- Review a random sample of generated episodes manually
- Confirm no obvious leakage or timestamp errors
- Save a small adjudicated audit set manually in `docs/episode_audit_notes.md`

## FOR CURSOR
**Prompt to Cursor:**
Implement deterministic episode generators for hyperkalemia, hypoglycemia, and sustained hypotension. Each generator must create positive and negative decision windows, derive protocol-based accepted action families, separately derive observed clinician actions from downstream records, and enforce strict no-future-leakage. Add audit utilities that print human-readable summaries for random sampled episodes.

## DONE CRITERIA
- Episode tables exist for all three tasks
- Positive/negative windows are reproducible
- Human-readable audit summaries look correct
- Split generation is deterministic

---

# Phase 6. Train/validation/test splitting

## Objective
Create reliable benchmark splits that prevent leakage across patient or stay.

## Split rules
- Split at patient level when possible
- No same patient across train/val/test
- Keep per-task distribution summaries
- Save fixed split files under version control or checksum them

## FOR ME
- Decide whether to use patient-level or admission-level split. Patient-level is safer.
- Freeze split seed and version

## FOR CURSOR
**Prompt to Cursor:**
Build deterministic train/validation/test split generation for benchmark episodes using patient-level separation. Output split statistics per task and per label, save split manifests, and add tests that prove no patient leakage across splits.

## DONE CRITERIA
- Split manifests saved
- No leakage
- Statistics exported

---

# Phase 7. Replay environment and structured tools

## Objective
Create the environment the agent will interact with.

## Tool philosophy
The agent must not directly inspect raw full tables. It must call structured tools.

## Required tools
Implement at least these:
- `get_recent_labs(stay_id, before_time, lab_names, hours_back)`
- `get_recent_vitals(stay_id, before_time, vital_names, minutes_back)`
- `get_recent_meds(hadm_id, before_time, med_classes, hours_back)`
- `get_recent_fluids(stay_id, before_time, hours_back)`
- `get_recent_procedures(stay_id, before_time, hours_back)`
- `get_trend(stay_id, signal_name, before_time, lookback, aggregation)`
- `run_task_calculator(task_name, context_json)`
- optional: `search_notes(hadm_id, before_time, query)`
- optional: `get_protocol_text(task_name)`

## Tool output rules
- JSON only
- Must include provenance
- Must include time ranges used
- Must not include future events

## FOR ME
- Decide whether notes are included in v1 or only as a later extension
- Decide which calculator functions are useful enough to justify inclusion

## FOR CURSOR
**Prompt to Cursor:**
Implement a replay environment with structured JSON-returning tools for labs, vitals, meds, fluids, procedures, trends, and task calculators. Every tool must enforce an as-of timestamp boundary and return provenance-rich results. Add test cases that intentionally try to access future data and confirm the environment blocks it.

## DONE CRITERIA
- Tool calls work on benchmark episodes
- Temporal boundary tests pass
- Output schema is stable

---

# Phase 8. Baselines before agent

## Objective
Build strong non-agent baselines before Gemini.

## Baseline A. Rule-only workflow
- Trigger fires from deterministic rules
- Returns predefined escalation template
- Minimal contextualization

## Baseline B. Tabular ML
- XGBoost or logistic regression
- Predict action family or escalation/no escalation from recent features
- Use structured features only

## Baseline C. Single-pass LLM
- Provide a compact structured summary of current state
- No tool use
- One call only

## Baseline D. Simple RAG
- Retrieve a fixed context pack from timeline and protocol text
- One-shot recommendation

## Common output schema
All baselines must emit the exact same JSON output schema as the agent:
- `trigger_type`
- `decision_time`
- `urgency_level`
- `recommended_next_steps`
- `evidence`
- `abstain`
- `confidence`
- `tool_trace`

## FOR ME
- Approve common schema before baseline implementation starts
- Decide whether the ML baseline predicts binary escalation or action family set. Prefer action family if feasible.

## FOR CURSOR
**Prompt to Cursor:**
Implement four comparable baselines: rule-only, XGBoost/logistic regression, single-pass LLM, and simple RAG. All baselines must share the same input episode format and output the same strict JSON schema as the final agent. Add evaluation hooks so all systems can be scored identically.

## DONE CRITERIA
- Baselines run end-to-end on sample episodes
- Outputs validate against the same schema
- Evaluation harness can score them

---

# Phase 9. Gemini integration and agent design

## Objective
Build the actual agentic system using Gemini API in a constrained, reproducible way.

## Recommended architecture
Use a small, disciplined pipeline, not a giant agent swarm.

### Module 1. Planner/Monitor
Input:
- task spec
- episode summary
- available tools

Output:
- ordered tool-use plan
- rationale fields in machine-readable form

### Module 2. Retriever/Executor
- deterministic Python executor for requested tools
- returns tool results

### Module 3. Verifier/Safety checker
- checks mandatory evidence coverage
- checks contradictions
- checks if action recommendation is unsupported
- can force abstention or defer decision

### Module 4. Reporter
- emits final JSON output only
- no narrative text in main run mode

## Gemini integration rules
- Use structured prompting
- Save all prompts and model outputs
- Keep temperature low
- Set max tool iterations, for example 3 to 5
- Log token usage and latency
- Keep model version fixed during benchmark runs

## Recommended Gemini tasks
Gemini should do:
- planning which tool to call next
- summarizing retrieved evidence into structured intermediate state
- deciding whether evidence is sufficient
- composing final structured report

Gemini should not do:
- direct raw SQL over uncontrolled tables
- free-form physician-style essays
- unbounded chain-of-thought logging into the paper

## FOR ME
1. Create Gemini API key in Google AI Studio or your chosen Gemini access point
2. Put it in `.env`
3. Record exact model name and version in `docs/model_registry.md`
4. Decide fixed generation settings for benchmark runs

## FOR CURSOR
**Prompt to Cursor:**
Integrate Gemini API into a constrained tool-using agent pipeline with four modules: planner, retriever/executor, verifier, and reporter. Use strict JSON schemas for all intermediate and final outputs. Add run logging for prompt text, tool calls, latency, and model metadata. Keep the agent deterministic where possible by fixing temperature and limiting iterations.

## DONE CRITERIA
- Gemini can run on one episode end-to-end
- All outputs are schema-valid
- Logs are saved
- Tool loop is bounded

---

# Phase 10. Output schemas and benchmark contract

## Objective
Define exactly what the model must output so evaluation is objective.

## Final required output schema
```json
{
  "episode_id": "string",
  "task_name": "hyperkalemia | hypoglycemia | hypotension",
  "trigger_detected": true,
  "trigger_type": "string",
  "decision_time": "ISO timestamp",
  "urgency_level": "low | medium | high | critical",
  "recommended_next_steps": ["string"],
  "recommended_action_families": ["string"],
  "evidence": [
    {
      "source_table": "string",
      "canonical_name": "string",
      "event_time": "ISO timestamp",
      "value": "string or number",
      "why_relevant": "string"
    }
  ],
  "missing_information": ["string"],
  "abstain": false,
  "abstain_reason": "string or null",
  "confidence": 0.0,
  "tool_trace": [
    {
      "tool_name": "string",
      "arguments": {},
      "returned_count": 0
    }
  ]
}
```

## FOR ME
- Approve this schema or modify once, then freeze it

## FOR CURSOR
**Prompt to Cursor:**
Define pydantic schemas for all benchmark inputs, intermediate states, tool results, and final outputs. Add schema validation in every runner path and fail loudly on malformed outputs.

## DONE CRITERIA
- Schema frozen
- Validation active everywhere

---

# Phase 11. Evaluation metrics

## Objective
Score not only final recommendation quality but also the agentic process.

## Primary metrics

### Trigger detection
- precision
- recall
- F1
- detection delay from first true trigger time

### Action recommendation
- exact match where appropriate
- set-overlap F1 / Jaccard against accepted action family set
- top-1 action family accuracy if using ranked outputs

### Evidence retrieval quality
- recall for mandatory evidence types
- evidence precision optional
- provenance correctness

### Safety
- false escalation rate in negative windows
- unsupported recommendation rate
- abstention appropriateness

### Efficiency / behavior
- number of tool calls
- latency
- tool plan length
- schema compliance rate

## Secondary analyses
- performance by task
- performance by ICU length of stay
- performance by missingness level
- performance with/without notes
- performance under noisy or partial context

## FOR ME
- Decide primary endpoint for paper abstract. Likely action-set F1 plus evidence recall and false escalation rate.

## FOR CURSOR
**Prompt to Cursor:**
Implement an evaluation suite for trigger detection, action recommendation agreement, evidence recall, safety metrics, abstention behavior, latency, and tool-use efficiency. Produce both aggregate and per-task reports, plus machine-readable outputs for plotting.

## DONE CRITERIA
- Evaluation runs on all systems
- Reports and plots generated

---

# Phase 12. Ablations and realism checks

## Objective
Prove the agentic parts matter.

## Required ablations
- no verifier
- no planner, fixed tool order
- no tools, prompt-only
- no notes
- no protocol text
- no abstention
- single-step vs multi-step tool budget

## Stress tests
- missing labs
- delayed measurements
- contradictory data
- partial context windows

## FOR ME
- Prioritize a minimal ablation set if compute or time is limited

## FOR CURSOR
**Prompt to Cursor:**
Add configurable ablations and stress tests for planner removal, verifier removal, tool removal, note removal, protocol text removal, abstention disabling, and varying tool budgets. Ensure the evaluation harness can run full-factorial or selected ablation subsets and save comparable outputs.

## DONE CRITERIA
- Ablations run reproducibly
- Results can support causal claims about agent components

---

# Phase 13. Error analysis

## Objective
Turn results into a paper, not just a benchmark dump.

## Error categories
- missed trigger
- late trigger detection
- wrong action family
- insufficient evidence retrieved
- evidence retrieved but ignored
- contradiction not caught
- unsafe escalation
- unnecessary abstention
- should abstain but did not

## FOR ME
- Manually inspect 20 to 40 failure cases
- Write notes per error mode in `docs/error_analysis_notes.md`

## FOR CURSOR
**Prompt to Cursor:**
Implement automatic error bucketing for failed benchmark episodes, and generate compact case reports with episode metadata, retrieved evidence, predicted actions, gold protocol action families, observed clinician actions, and likely failure category.

## DONE CRITERIA
- Failure cases are easy to inspect
- Error categories support the paper narrative

---

# Phase 14. Benchmark packaging and reproducibility

## Objective
Package the benchmark so it is publishable and reusable.

## Required artifacts
- task specs
- codebooks
- split manifests
- benchmark episode files
- replay environment code
- baseline runners
- agent runner
- evaluation scripts
- config examples
- reproduction instructions

## FOR ME
- Decide what can be released based on MIMIC licensing constraints
- Prepare a reproducibility checklist

## FOR CURSOR
**Prompt to Cursor:**
Create release-ready packaging for the benchmark and system: CLI entrypoints, config templates, reproduction scripts, environment files, and documentation for generating benchmark episodes from licensed local MIMIC data.

## DONE CRITERIA
- Another licensed researcher could reproduce the pipeline from docs

---

# Phase 15. Suggested execution order for Cursor

Use these prompts in sequence. Do not skip ahead.

## Cursor Prompt 1
Build the repository skeleton, environment files, tests, and config-driven local MIMIC data access layer. Support offline CSV/Parquet or PostgreSQL. Add a generated inventory report.

## Cursor Prompt 2
Implement task-spec YAML files and validation for hyperkalemia, hypoglycemia, and sustained hypotension, with thresholds, action windows, exclusions, and required evidence types.

## Cursor Prompt 3
Implement normalization codebooks and pipelines for required labs, vitals, medications, fluids, and procedures. Flag ambiguous mappings and add coverage tests.

## Cursor Prompt 4
Build the canonical timeline generator with strict provenance and as-of-time slicing.

## Cursor Prompt 5
Implement deterministic episode generation with positive/negative windows, protocol action labels, observed clinician action labels, and no-future-leakage tests.

## Cursor Prompt 6
Implement deterministic patient-level train/validation/test splits and save manifests and statistics.

## Cursor Prompt 7
Build the replay environment and structured tools with timestamp-bounded access and JSON outputs.

## Cursor Prompt 8
Implement rule, tabular ML, single-pass LLM, and RAG baselines with the same output schema.

## Cursor Prompt 9
Integrate Gemini API into a planner -> retriever/executor -> verifier -> reporter pipeline with strict schemas and logging.

## Cursor Prompt 10
Implement evaluation, ablations, stress tests, and automatic error bucketing.

## Cursor Prompt 11
Package the benchmark and create reproducibility docs and CLI entrypoints.

---

# Phase 16. Immediate next actions this week

## FOR ME, day 1
- Put MIMIC data on disk
- Confirm table names
- Create `.env`
- Freeze task scope

## FOR CURSOR, day 1
Use Cursor Prompt 1 only

## FOR ME, day 2
- Review generated inventory report
- Check missing tables
- Inspect a few raw rows for potassium, glucose, MAP, insulin, dextrose, vasopressors, calcium, dialysis

## FOR CURSOR, day 2
Use Cursor Prompt 2 and Prompt 3

## FOR ME, day 3
- Approve trigger definitions
- Approve codebook corrections

## FOR CURSOR, day 3 to 4
Use Cursor Prompt 4 and Prompt 5

## FOR ME, day 5
- Audit sample episodes manually
- Check label logic

## FOR CURSOR, day 5 to 6
Use Cursor Prompt 6 and Prompt 7

## FOR ME, day 7
- Approve replay environment outputs
- Decide whether notes are in or postponed

---

# Common failure modes to avoid

1. Building the agent before the labels are correct
2. Mixing observed clinician behavior with gold protocol label
3. Allowing future leakage through trend computation or note retrieval
4. Using vague free-text outputs that cannot be scored
5. Choosing too many tasks too early
6. Claiming clinical correctness beyond the benchmark rules
7. Ignoring action provenance and evidence provenance
8. Using notes too early before structured tasks are stable

---

# Minimal benchmark claims the paper can safely make

- We introduce a reproducible event-triggered ICU escalation benchmark on public MIMIC-IV data.
- The benchmark evaluates not only prediction quality but also evidence retrieval, tool use, abstention, and structured action recommendation.
- A tool-using agent outperforms non-agent baselines on workflow-level metrics such as evidence completeness and action-set quality.

Avoid claiming:
- real clinical deployment readiness
- physician-level equivalence
- globally correct clinical management beyond task-specific protocol labels

---

# Final note
The benchmark is the paper’s backbone. If the benchmark is clean, even a modest agent can become publishable. If the benchmark is weak, even a fancy agent will not save the paper.
