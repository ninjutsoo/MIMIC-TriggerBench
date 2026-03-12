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

These future-looking conditions are **label-generation-only** constructs. They are permitted when defining which decision times count as negatives but **must never be exposed** in replay tools, episode context passed to the agent, or any tool outputs. Downstream components may only see as-of-time information even when episodes were sampled using future-based criteria.

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

## Mapping contract (deterministic and auditable)
Maintain a row-level mapping ledger for every candidate source concept, including excluded concepts.

Required mapping fields:
- `source_table`
- `source_identifier` (for example `itemid`, medication name, or procedure code)
- `source_label`
- `mapping_decision` with explicit statuses (for example `map_as_is`, `map_convert_unit`, `no_mapping`, `unsure`)
- `canonical_concept`
- `target_unit` and explicit conversion parameters when conversion is required
- optional standard vocabulary fields when available (`loinc_code`, `rxnorm_code`, `snomed_code`)
- `source_frequency_count` and unit-frequency summary used during prioritization
- `review_status` and `review_note`

Normalization rules:
- Prefer identifier-based mapping over free-text-only matching when both are available.
- Keep `no_mapping` and `unsure` entries explicit; do not silently drop them.
- Treat unit conversion rules as versioned mapping logic, not ad hoc code in downstream detectors.

## Direct quote anchors
- `"itemid (omop_source_code),label,omop_concept_id,omop_concept_name,omop_domain_id,omop_vocabulary_id,omop_concept_class_id,omop_standard_concept,omop_concept_code,sssom_author_id,sssom_reviewer_id,sssom_subject_source_version,common_vocabulary_version,sssom_mapping_tool,sssom_mapping_date,sssom_confidence,sssom_comment"`
- `"itemid (omop_source_code),label,ordercategorydescription,amountuom,omop_concept_id,omop_concept_name,omop_domain_id,omop_vocabulary_id,omop_concept_class_id,omop_standard_concept,omop_concept_code,sssom_author_id,sssom_reviewer_id,sssom_subject_source_version,sssom_mapping_tool,sssom_mapping_date,sssom_confidence,sssom_comment,inputevents_row_count"`
- `"lab_category,decision,itemid,label,abbreviation,fluid,category,count,value_instances,uom_instances,target_uom,conversion_multiplier,status,note"`
- `"albumin,\"TO MAP, AS IS\",50862,Albumin,,Blood,Chemistry,749944"`
- `"albumin,NO MAPPING,51069,\"Albumin, Urine\",,Urine,Chemistry,56977"`
- `"chloride,\"TO MAP, CONVERT UOM\",50902,Chloride,,Blood,Chemistry,3083705"`
- `"med_category,decision,itemid,label,abbreviation,linksto,category,unitname,param_type,count,value_instances,amountuom_instances,rateuom_instances,ordercategoryname_instances,secondaryordercategoryname_instances,ordercategorydescription_instances,note,status,reviewer"`
- `"norepinephrine,CONTINUOUS,221906,Norepinephrine,Norepinephrine,inputevents,Medications,mg,Solution,459800"`
- `"heparin,UNSURE,224145,Heparin Dose (per hour),Heparin Dose (per hour),chartevents,Dialysis,units,Numeric,181081"`
- `"expression\" : \"(value.exists() or valuenum.exists()) and valueuom.exists() and valueuom.trim() != ''\""`
- `"We need to create a concept_id for each MIMIC-III local code."`
- `"\"regex\": \"aztreonam|bactrim|cephalexin|chloramphenicol|cipro|flagyl|metronidazole|nitrofurantoin|tazobactam|rifampin|sulfadiazine|timentin|trimethoprim|(amika|gentami|vanco)cin|(amoxi|ampi|dicloxa|naf|oxa|peni|pipera)cillin|(azithro|clarithro|erythro|clinda|strepto|tobra|vanco)mycin|cef(azolin|tazidime|adroxil|epime|otetan|otaxime|podoxime|uroxime)|(doxy|mino|tetra)cycline|(levofl|moxifl|ofl)oxacin|macro(bid|dantin)|(una|zo)syn\""`
- `"\"table\": \"inputevents\", \"sub_var\": \"itemid\", \"callback\": \"transform_fun(set_val(TRUE))\""`

## FOR ME
- Inspect a sample of raw MIMIC rows for each target concept
- Manually verify ambiguous medication names and ICU item mappings
- Log unresolved ambiguities instead of silently forcing mappings

## FOR CURSOR
**Prompt to Cursor:**
Build normalization pipelines and codebooks for labs, hemodynamics, medications, fluids, and procedures required by the three benchmark tasks. The system must map raw MIMIC labels/itemids/drug names into canonical concepts, attach units, flag ambiguous rows, and include tests for mapping coverage and unexpected unmapped high-frequency terms. Implement a row-level mapping ledger with explicit mapping decision statuses, conversion parameters, source-frequency summaries, and review metadata. Implement the direct quote anchors in this phase and document any intentional deviations.

## DONE CRITERIA
- Canonical codebooks exist
- High-frequency rows are mapped or explicitly flagged
- Unit normalization is tested
- Ambiguous mappings are tracked
- Mapping rows retain explicit decision status and review/provenance metadata

---

# Phase 3.5. Action extraction feasibility checkpoint

## Objective
Confirm that normalized events are rich enough to reliably detect the action patterns that define gold labels before proceeding to episode and label generation.

## Required feasibility checks
- Insulin + dextrose pairing within clinically reasonable windows for hyperkalemia and hypoglycemia tasks
- Vasopressor starts and dose escalations for sustained hypotension
- Crystalloid fluid boluses meeting the benchmark bolus definition
- Dialysis or RRT starts relevant to hyperkalemia management

## Feasibility acceptance rules (gate to Phase 5)
- Build detector outputs as auditable event tables, not only aggregate counts.
- For infusion-based actions, normalize dose/rate units before start/escalation logic.
- For each action family, report at least:
  - number of detected events,
  - number of unique stays/patients,
  - train/validation/test support after splitting,
  - detector failure patterns (missing fields, ambiguous routes, conflicting units).
- Run manual chart-level review on sampled detections and sampled non-detections for each action family (use all cases if very small, otherwise at least 25 per sample type).
- Freeze explicit go/no-go thresholds before running the checkpoint (for example minimum support and minimum manual-review precision), and enforce that unsupported action families are down-scoped before Phase 5.

## Direct quote anchors
- `"CASE WHEN rateuom = 'mg/kg/min' AND patientweight = 1 THEN rate"`
- `"WHEN rateuom = 'mg/kg/min' THEN rate * 1000.0"`
- `"ELSE rate END AS vaso_rate"`
- `"CASE WHEN rateuom = 'units/min' THEN rate * 60.0 ELSE rate END AS vaso_rate"`
- `"LEAD(vasotime, 1) OVER (PARTITION BY stay_id ORDER BY vasotime) AS endtime"`
- `"WHERE t.endtime IS NOT NULL;"`
- `"excluded_labels = [\"NO MAPPING\", \"UNSURE\", \"NOT AVAILABLE\"]"`
- `"WHEN _item_class = 'INTERMITTENT' OR (_item_class = 'BOTH' AND ({find_intm_where_clause})) THEN 'intm'"`
- `"WHEN _item_class = 'CONTINUOUS' OR (_item_class = 'BOTH' AND NOT ({find_intm_where_clause})) THEN 'cont'"`
- `"ORDER BY hadm_id, starttime, linkorderid, med_category, endtime"`
- `"CLIF_CRRT_SCHEMA = pa.DataFrameSchema("`
- `"CAST(blood_flow_rate as FLOAT) * 60 as blood_flow_rate"`
- `"CLIF_CRRT_SCHEMA.validate(crrt_events_cast_and_cleaned, lazy=True)"`

## FOR ME
- Review small, manually inspected samples for each action pattern
- Decide whether observability is sufficient for each action family or whether certain actions need to be down-scoped or dropped from the v1 protocol label set
- Record findings and any scope adjustments in `docs/trigger_definitions.md` and `docs/episode_audit_notes.md`

## FOR CURSOR
**Prompt to Cursor:**
Using the normalized codebooks and pipelines from Phase 3, build deterministic detectors and coverage reports for insulin+dextrose pairings, vasopressor starts/escalations, fluid boluses, and dialysis starts. For infusion-based detectors, normalize rates before start/escalation logic. Output both row-level detector tables and aggregate summaries, plus sampled positive/negative review sets for feasibility decisions. Implement the direct quote anchors in this phase and document any intentional deviations.

## DONE CRITERIA
- Coverage reports exist for each targeted action family
- At least a few manually reviewed examples per action type are documented
- A go/no-go decision is recorded on whether each action family is reliable enough to be part of protocol-derived gold labels in Phase 5
- Pre-frozen feasibility thresholds are either met or the action family is explicitly down-scoped

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
- `event_time_end` nullable
- `event_uid` deterministic identifier
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
## Direct quote anchors
- `"code: [PROCEDURE, START, col(itemid)]"` and `"code: [PROCEDURE, END, col(itemid)]"`
- `"code: [INFUSION_START, col(itemid)]"` and `"code: [INFUSION_END, col(itemid)]"`
- `"order_id: orderid"` and `"link_order_id: linkorderid"`
- `"uuid_generate_v5(ns_medication_administration_icu.uuid, ie.stay_id || '-' || ie.orderid || '-' || ie.itemid) AS uuid_INPUTEVENT"`
- `"LEAD(vasotime, 1) OVER (PARTITION BY stay_id ORDER BY vasotime) AS endtime"`
- `"WHERE t.endtime IS NOT NULL;"`

- Preserve source provenance
- Preserve exact timestamps
- Support multiple events with same timestamp
- Enforce deterministic tie-break ordering for same-timestamp events using a fixed sort key
- Preserve interval semantics for start/end events (for example infusions/procedures)
- Store mapping/version provenance in `metadata_json`
- Support filtering by event category and canonical name
- Support “as-of time” slicing for replay

## FOR ME
- Decide output storage format for canonical timelines, ideally Parquet
- Confirm enough disk space for derived tables

## FOR CURSOR
**Prompt to Cursor:**
Implement a canonical timeline builder that merges normalized events from required MIMIC tables into a single time-ordered representation per ICU stay. Include deterministic `event_uid` generation and a fixed tie-break sort key for same-timestamp events. Preserve interval semantics (`event_time` and nullable `event_time_end`), save outputs as partitioned Parquet by task-relevant cohorts, and include indexing utilities for fast as-of time slicing plus tests for ordering determinism and provenance preservation. Implement the direct quote anchors in this phase and document any intentional deviations.

## DONE CRITERIA
- Canonical timelines exist for sample stays
- As-of slicing works
- Tests verify strict ordering and provenance
- Same-input rebuilds produce identical event ordering and `event_uid` values

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
Implement deterministic episode generators for hyperkalemia, hypoglycemia, and sustained hypotension. Each generator must:
- create positive and negative decision windows (using the negative window definition from Phase 2),
- derive protocol-based accepted action families,
- separately derive observed clinician actions from downstream records,
- and enforce strict no-future-leakage.
Ensure that any future-looking logic used only for defining negative windows is not surfaced in episode context or replay tools. Add audit utilities that print human-readable summaries for random sampled episodes and a regression test that asserts negative-window-only fields are never accessible through replay or tool interfaces.

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

# Phase 6.5. Output schemas and runner validation gate

## Objective
Freeze the final benchmark output schema and wire strict validation into all runners **before** building replay tools, baselines, or the agent.

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
- Approve this schema or modify once, then freeze it as the v1 contract

## FOR CURSOR
**Prompt to Cursor:**
Define pydantic schemas for all benchmark inputs, intermediate states, tool results, and final outputs. Add schema validation in every runner path (replay tools, baselines, agent) and fail loudly on malformed outputs. Treat this schema as frozen for all later phases.

## DONE CRITERIA
- Schema frozen prior to Phase 7 work
- Validation active in all runners used in later phases

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
- Output schema matches the frozen schema from Phase 6.5

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

## Feature builder contract for tabular ML
- Define a dedicated, deterministic feature-table builder module that consumes canonical timelines and episode definitions, applies fixed lookback windows/bins/aggregation rules, and outputs a stable feature table schema with named columns and types.
- Enforce split-aware fit/transform behavior: any learned preprocessing for the tabular baseline (imputation values, scaling, clipping, encoding) must be fit on train only and reused unchanged for validation/test.
- Include explicit missingness and measurement-count features alongside value aggregates.
- Persist `feature_spec_version`, config hash, and split seed with generated feature artifacts.
- This module should be the **only** way tabular baselines obtain features, so that experiments are reproducible and comparable.

## Direct quote anchors
- `"Fits and transforms the training features, then transforms the validation and test features with the recipe."`
- `"data[DataSplit.train][type] = recipe.prep()"`
- `"data[DataSplit.val][type] = recipe.bake(data[DataSplit.val][type])"`
- `"data[DataSplit.test][type] = recipe.bake(data[DataSplit.test][type])"`
- `"outer_cv = StratifiedShuffleSplit(cv_repetitions, train_size=train_size, random_state=seed)"`
- `"inner_cv = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)"`
- `"sta_rec.add_step(StepSklearn(MissingIndicator(features=\"all\"), sel=all_of(vars[DataSegment.static]), in_place=False))"`
- `"data.add_step(StepHistorical(sel=dynamic_vars, fun=Accumulator.COUNT, suffix=\"count_hist\"))"`
- `"('temperature',              'f',   lambda x: x > 79, lambda x: (x - 32) * 5./9),"`
- `"X = X.groupby(ID_COLS + group_item_cols + ['hours_in']).agg(['mean', 'std', 'count'])"`

## Common output schema
All baselines must emit the exact same full JSON output schema as the agent, matching the frozen schema from Phase 6.5. The field list shown there is the authoritative contract; do not implement a baseline-specific subset.

## FOR ME
- Approve common schema before baseline implementation starts
- Decide whether the ML baseline predicts binary escalation or action family set. Prefer action family if feasible.

## FOR CURSOR
**Prompt to Cursor:**
Implement four comparable baselines: rule-only, XGBoost/logistic regression, single-pass LLM, and simple RAG. All baselines must:
- share the same input episode format,
- obtain features for the tabular model only through the dedicated feature-table builder,
- enforce train-only fitting for any learned preprocessing in the tabular path,
- and output the same strict JSON schema as the final agent, as defined in Phase 6.5.
Add evaluation hooks so all systems can be scored identically, and write feature artifact metadata including feature spec version/hash and split seed.
Implement the direct quote anchors in this phase and document any intentional deviations.

## DONE CRITERIA
- Baselines run end-to-end on sample episodes
- Outputs validate against the same schema
- Evaluation harness can score them
- Feature artifacts include reproducible feature-spec metadata and train-only preprocessing auditability

---

# Phase 9. Gemini integration and agent design

## Objective
Build the actual agentic system using Gemini API in a constrained, reproducible way.

## Recommended architecture
Use a small, disciplined pipeline, not a giant agent swarm. In code, this should look like:

- a **single bounded agent loop** that:
  - receives the task spec, episode summary, and available tools,
  - lets Gemini plan which tool to call next (up to a small fixed budget of iterations),
  - executes tools via a deterministic Python dispatcher,
  - accumulates a structured intermediate state, and
  - asks Gemini once more to emit the final JSON report;
- plus a **deterministic verifier/safety checker** that:
  - checks mandatory evidence coverage,
  - checks contradictions,
  - checks if any recommended action is unsupported,
  - and can force abstention or defer the decision.

Conceptually, this covers planner, retriever/executor, verifier, and reporter roles, but it should be implemented as one tight loop plus a verifier, not four heavyweight subsystems.

## Gemini integration rules
- Use structured prompting
- Save all prompts and model outputs
- Keep temperature low
- Set max tool iterations, for example 3 to 5
- Log token usage and latency
- Keep model version fixed during benchmark runs
 - Use direct Gemini API/SDK calls with pydantic models and a small Python tool dispatcher; avoid heavy agent frameworks (e.g., LangChain) in v1 unless a concrete benchmark-driven need is identified later.

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
Integrate Gemini API into a constrained tool-using agent pipeline implemented as a single bounded planner/executor loop plus a deterministic verifier and final reporter. Use direct Gemini API/SDK calls, pydantic models, and a small Python tool dispatcher; do not introduce LangChain or similar heavy agent frameworks in v1. Use strict JSON schemas for all intermediate and final outputs. Add run logging for prompt text, tool calls, latency, and model metadata. Keep the agent deterministic where possible by fixing temperature and limiting iterations.

## DONE CRITERIA
- Gemini can run on one episode end-to-end
- All outputs are schema-valid
- Logs are saved
- Tool loop is bounded

---

# Phase 10. Evaluation metrics

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

# Phase 11. Ablations and realism checks

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

# Phase 12. Error analysis

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

# Phase 13. Benchmark packaging and reproducibility

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
- reproducibility manifest with pinned dataset version, mapping/codebook version, schema version, split seed, feature spec hash, and model identifiers

## Direct quote anchors
- `"event_conversion_config_fp: ${oc.env:EVENT_CONVERSION_CONFIG_FP}"`
- `"etl_metadata: dataset_name: ${oc.env:DATASET_NAME}"` and `"dataset_version: ${oc.env:DATASET_VERSION}"`
- `"expected_files = { 'dataset.json', 'codes.parquet', 'subject_splits.parquet' }"`
- `"root_output_dir: ???"`, `"pre_MEDS_dir: ${root_output_dir}/pre_MEDS"`, `"MEDS_cohort_dir: ${root_output_dir}/MEDS_cohort"`
- `"hydra:"`
- `"run: dir: ${log_dir}"`
- `"sweep: dir: ${log_dir}"`

## FOR ME
- Decide what can be released based on MIMIC licensing constraints
- Prepare a reproducibility checklist

## FOR CURSOR
**Prompt to Cursor:**
Create release-ready packaging for the benchmark and system: CLI entrypoints, config templates, reproduction scripts, environment files, and documentation for generating benchmark episodes from licensed local MIMIC data. Emit a machine-readable reproducibility manifest that pins dataset version, mapping/schema versions, split seed, feature spec hash, and model IDs used in each benchmark run. Implement the direct quote anchors in this phase and document any intentional deviations.

## DONE CRITERIA
- Another licensed researcher could reproduce the pipeline from docs
- Reproducibility manifest is generated and sufficient to rerun the same benchmark configuration

---

# Phase 14. Suggested execution order for Cursor

Use these prompts in sequence. Do not skip ahead.

## Cursor Prompt 1
Build the repository skeleton, environment files, tests, and config-driven local MIMIC data access layer. Support offline CSV/Parquet or PostgreSQL. Add a generated inventory report.

## Cursor Prompt 2
Implement task-spec YAML files and validation for hyperkalemia, hypoglycemia, and sustained hypotension, with thresholds, action windows, exclusions, and required evidence types.

## Cursor Prompt 3
Implement normalization codebooks and pipelines for required labs, vitals, medications, fluids, and procedures. Include a row-level mapping ledger with explicit decision status, conversion parameters, source-frequency summaries, and review metadata. Flag ambiguous mappings and add coverage tests.

## Cursor Prompt 4
Implement the action extraction feasibility checkpoint: build deterministic detectors and coverage reports for insulin+dextrose pairings, vasopressor starts/escalations, fluid boluses, and dialysis starts. Output row-level detector tables plus aggregate support metrics and sampled manual-review sets, then document whether each candidate action family is reliable enough to remain in the protocol-derived gold label set.

## Cursor Prompt 5
Build the canonical timeline generator with strict provenance and as-of-time slicing, including deterministic `event_uid` values, stable same-timestamp ordering, and interval-event handling.

## Cursor Prompt 6
Implement deterministic episode generation with positive/negative windows, protocol action labels, observed clinician action labels, and no-future-leakage tests.

## Cursor Prompt 7
Implement deterministic patient-level train/validation/test splits and save manifests and statistics.

## Cursor Prompt 8
Freeze the shared benchmark JSON schemas and add validation for benchmark inputs, tool results, intermediate states, and final outputs before building replay tools, baselines, or the agent.

## Cursor Prompt 9
Build the replay environment and structured tools with timestamp-bounded access and JSON outputs.

## Cursor Prompt 10
Implement rule, tabular ML, single-pass LLM, and RAG baselines with the same frozen output schema. The tabular ML baseline must obtain features only through the dedicated deterministic feature-table builder, enforce train-only fitting for learned preprocessing, and persist feature-spec hash metadata.

## Cursor Prompt 11
Integrate Gemini API into a bounded planner/executor loop plus deterministic verifier/reporter pipeline with strict schemas and logging, using direct Gemini API/SDK calls and no LangChain in v1.

## Cursor Prompt 12
Implement evaluation, ablations, stress tests, and automatic error bucketing.

## Cursor Prompt 13
Package the benchmark and create reproducibility docs and CLI entrypoints, including a machine-readable reproducibility manifest for each run configuration.

---

# Phase 15. Immediate next actions this week

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
- Review action-extraction feasibility samples and decide whether any action families need to be narrowed or dropped

## FOR CURSOR, day 3 to 4
Use Cursor Prompt 4 and Prompt 5

## FOR ME, day 5
- Audit sample episodes manually
- Check label logic
- Approve the frozen benchmark schema before replay tools and model runners are built

## FOR CURSOR, day 5 to 6
Use Cursor Prompt 6, Prompt 7, and Prompt 8

## FOR ME, day 7
- Approve replay environment outputs
- Decide whether notes are in or postponed

## FOR CURSOR, day 7
Use Cursor Prompt 9 only

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
