# PAPER_TASKS.md

## Project
Paper plan for MIMIC-TriggerBench: Event-Triggered ICU Monitoring and Escalation Agents on MIMIC-IV

## Goal
Write the paper in parallel with implementation so the project stays aligned with publishable claims, clear baselines, and rigorous literature positioning.

This document separates:
- **FOR ME**: manual research, reading, writing, figure planning
- **FOR CHATGPT / DEEP RESEARCH**: exact prompts to use
- **FOR CURSOR**: code/docs generation tasks that support writing

The paper should be benchmark-first, agent-second.

---

# Paper structure

Target default structure:
1. Introduction
2. Related Work
3. Benchmark and Task Definition
4. Data and Label Construction
5. Agent and Baselines
6. Evaluation Protocol
7. Results
8. Ablations and Error Analysis
9. Limitations, Ethics, and Reproducibility
10. Conclusion

---

# Phase A. Literature review and positioning

## Objective
Build a complete literature map so the paper does not overclaim novelty.

## Core literature buckets to cover
1. Agentic AI in healthcare surveys
2. Medical EHR agent benchmarks
3. Multi-agent clinical reasoning systems
4. ICU deterioration and early-warning models on MIMIC
5. Clinical protocol or workflow automation systems
6. Tool-using medical LLM systems
7. Evidence-grounded or traceable clinical AI systems

## FOR ME
- Create a spreadsheet or markdown table with columns:
  - paper
  - venue
  - year
  - task
  - dataset
  - agentic features
  - evaluation
  - why not the same as our work
- Maintain a `refs/` folder with PDFs and notes

## Prompt for ChatGPT Deep Research: literature completeness
Use this in Deep Research mode:

```text
Find papers from 2023 to 2026 on agentic AI systems in healthcare or medicine, especially systems involving EHR interaction, ICU monitoring, clinical workflow automation, or tool-using medical LLMs. Search at least these venues: NeurIPS, ICML, ICLR, AAAI, KDD, CHIL, MIDL, MLHC, Nature Digital Medicine, NEJM AI, JAMIA, JMIR AI. Include only papers where the system is genuinely agentic or workflow-based, not simple classification. For each paper give: title, venue, year, dataset, task, whether it uses planning/tool use/multi-step reasoning, evaluation protocol, and why it is or is not close to our proposed work: an event-triggered ICU escalation benchmark and tool-using agent on MIMIC-IV. Mark the closest works and explicitly say what novelty remains for our project.
```

## Prompt for ChatGPT normal search mode: focused related work table
```text
Build me a concise related-work table for a paper on event-triggered ICU monitoring and escalation agents on MIMIC-IV. Focus on: MedAgentBench, ColaCare, TriageAgent, EHR-navigation agents, ICU deterioration models on MIMIC, and tool-using medical LLMs. For each, provide one sentence for what they do, one sentence for why they do not solve our exact problem, and a citation-ready venue/year line.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/related_work_table.md` and `paper/bib_candidates.md` templates with columns for task, dataset, agent design, evaluation, and novelty relative to our benchmark. Also create a `paper/claims_to_verify.md` file listing every claim that will require citation.

## DONE CRITERIA
- Literature table has at least 20 to 40 serious papers
- Closest prior works are explicitly compared against our project
- Novelty statement is narrow and defensible

---

# Phase B. Introduction drafting early

## Objective
Write the introduction early so it constrains implementation.

## What the introduction must say
- Current medical agents often show tool use or interaction, but not event-triggered ICU workflow replay over public MIMIC ICU streams
- Existing systems under-emphasize workflow metrics such as trigger timeliness, evidence completeness, abstention, and traceable action recommendation
- We introduce a benchmark and a tool-using agent for event-triggered escalation tasks with deterministic labels on public data

## FOR ME
Draft the introduction in three claims only:
1. why the problem matters
2. what is missing
3. what we contribute

## Prompt for ChatGPT normal mode: intro draft
```text
Draft a rigorous NeurIPS-style introduction for a paper on MIMIC-TriggerBench, a benchmark for event-triggered ICU escalation agents on MIMIC-IV. Keep the novelty narrow and defensible. Do not overclaim physician-level reasoning. Emphasize public-data reproducibility, benchmark design, workflow-level evaluation, and why tool-using agents are more appropriate than classifiers for this task.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/01_introduction.md` with placeholders for cited claims, contribution bullets, and a problem framing paragraph that matches a benchmark-first paper.

## DONE CRITERIA
- Intro skeleton exists
- Contribution bullets are aligned with actual implementation

---

# Phase C. Related work writing

## Objective
Write related work as comparison, not summary.

## Sections to include
1. Agentic medical AI systems
2. EHR and interactive medical benchmarks
3. ICU monitoring and deterioration prediction
4. Tool-using LLM systems and traceable clinical AI

## FOR ME
For each section, write:
- one paragraph on what exists
- one paragraph on what remains unsolved for our benchmark

## Prompt for ChatGPT Deep Research: gap verification
```text
Given our proposed project, identify the closest prior works and explicitly test whether any of them already implements the same combination of properties: event-triggered ICU streaming, MIMIC-IV ICU tables, structured tool use during replay, deterministic trigger/action labels, and workflow-level evaluation of evidence retrieval plus escalation recommendations. If any prior work fully matches this, explain how and why our novelty would fail. If not, state the residual gap precisely.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/02_related_work.md` with subsection headers and fillable bullet placeholders under each subsection for: what prior work did, dataset, evaluation, and why it differs from our setting.

## DONE CRITERIA
- Related work is framed around differences that matter to reviewers
- Overlap risk is explicitly addressed

---

# Phase D. Benchmark and task-definition section

## Objective
This is the core section. It must be extremely precise.

## Must include
- overall replay setup
- task families
- trigger rules
- action windows
- output schema
- why tasks require agentic behavior
- what public tables are used

## FOR ME
- Keep a paper-ready task definition document synced with implementation
- Add one worked example episode diagram
- Track clinical trigger references here for later citation:
  - Hypotension: Solares et al., 2024, “The Value of the Hypotension Prediction Index as a Tool for Treatment of Haemodynamic Instability: A Clinical Review” (HPI / AQI metric with MAP < 65 mmHg for >= 15 minutes).
  - Hypotension outcomes: Salmasi et al., 2017, “Relationship between Intraoperative Hypotension, Defined by Either Reduction from Baseline or Absolute Thresholds, and Acute Kidney and Myocardial Injury” (risk rises sharply after ~13–15 minutes below MAP 65 mmHg).
  - Hypoglycemia: American Diabetes Association, 2025, “6. Glycemic Goals and Hypoglycemia: Standards of Care in Diabetes—2025” (Level 2 hypoglycemia < 54 mg/dL).
  - Hyperkalemia: UK Kidney Association, 2023, “Clinical Practice Guidelines: Treatment of Acute Hyperkalaemia in Adults” (mild 5.5–5.9, moderate 6.0–6.4, severe >= 6.5 mmol/L; our benchmark uses >= 6.0 mmol/L as a clinically important, moderate-to-severe trigger).
  - Hypoglycemia treatment nuance: Randomized ED evidence comparing D10/D25/D50 dextrose concentrations for hypoglycemia treatment (supports allowing D10 bolus vs only D50 in deterministic action labels when mapped from real-world records).
  - Hypotension episode structure: Evidence that hypotension occurs in temporal clusters during vasopressor infusion (supports explicit episode clustering / washout rules in the benchmark task spec).

## Prompt for ChatGPT normal mode: benchmark wording
```text
Write a benchmark section for a paper introducing an event-triggered ICU escalation benchmark on MIMIC-IV. Include a formal definition of episode replay, trigger events, agent inputs, structured tool interactions, output schema, and why the benchmark evaluates workflow execution rather than only prediction.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/03_benchmark_definition.md` and auto-generate a task summary table from the YAML task specs, including trigger rule, action window, evidence types, and output fields.

## DONE CRITERIA
- Benchmark section can be mostly filled directly from implementation artifacts

---

# Phase E. Data and label construction section

## Objective
Convince reviewers the labels are objective and leakage-free.

## Must include
- exact MIMIC tables used
- codebook construction
- canonical timeline generation
- episode generation
- split generation
- no-future-leakage policy
- protocol-action labels vs observed clinician actions

## FOR ME
- Write this section after manually auditing sampled episodes
- Include 2 to 3 concrete examples with timestamps

## Prompt for ChatGPT normal mode: label-method section
```text
Write a rigorous methods section explaining deterministic label construction for three event-triggered ICU tasks on MIMIC-IV: moderate-to-severe hyperkalemia (K >= 6.0 mmol/L), severe (Level 2) hypoglycemia, and sustained hypotension. Explain trigger labels, protocol-derived action labels, observed downstream clinician action labels, negative windows, and how future leakage is prevented.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/04_data_and_labels.md` with placeholders that can be auto-filled from code-generated dataset statistics, split summaries, and task-spec metadata. Also generate a table template listing all MIMIC tables and what they are used for.

## DONE CRITERIA
- Section explicitly separates gold protocol labels from observed clinician actions
- Leakage prevention is clearly documented

---

# Phase F. System and baselines section

## Objective
Present the agent architecture cleanly and keep baseline comparisons fair.

## Must include
- planner
- retriever/executor tools
- verifier/safety checker
- reporter
- rule baseline
- tabular ML baseline
- single-pass LLM baseline
- simple RAG baseline

## FOR ME
- Draw one system figure and one baseline comparison table

## Prompt for ChatGPT normal mode: method section for system
```text
Write a methods section for a medical benchmark paper describing a constrained tool-using agent pipeline with modules for planning, evidence retrieval, verification, and structured reporting. Include why each module exists, how it differs from a one-shot LLM or simple RAG baseline, and how the same output schema is enforced across all baselines.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/05_system_and_baselines.md` with placeholders for system description, tool list, model settings, and a comparison table of all baselines and their capabilities.

## DONE CRITERIA
- The system section mirrors the actual code paths
- Baseline table is fair and explicit

---

# Phase G. Evaluation section

## Objective
Make evaluation look strong and multi-dimensional.

## Must include
- primary metrics
- secondary metrics
- ablations
- stress tests
- significance tests if appropriate
- compute and API budget reporting if possible

## FOR ME
- Choose one headline metric bundle for abstract and main table
- Decide whether to include confidence intervals or bootstrap estimates

## Prompt for ChatGPT normal mode: evaluation section
```text
Write an evaluation section for a benchmark paper where models are compared on trigger detection, action recommendation quality, evidence retrieval recall, abstention safety, and tool-use efficiency. Emphasize why AUROC alone is insufficient for this benchmark.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/06_evaluation.md` with placeholders for metrics, baselines, splits, implementation details, and ablation settings. Also create scripts that export paper-ready result tables as CSV and markdown.

## DONE CRITERIA
- Evaluation section matches actual metrics in the code

---

# Phase H. Results, ablations, and error analysis

## Objective
Make the main empirical story sharp.

## Main story options
Option 1:
- Agents match or beat baselines on action-set quality
- Agents clearly outperform on evidence recall and abstention behavior

Option 2:
- Agents do not dominate on all metrics, but are much better on workflow metrics that matter for escalation

Either is acceptable if written honestly.

## FOR ME
- Write down the intended main claim only after seeing results
- Do not pre-commit to “agent beats everything”

## Prompt for ChatGPT normal mode: results narrative
```text
Given benchmark results comparing rule-based, tabular ML, one-shot LLM, RAG, and a tool-using agent, help me write a results narrative that is rigorous and honest. The narrative should focus on where the agent helps most: evidence completeness, action grounding, abstention, and workflow execution, even if it does not dominate every metric.
```

## Prompt for ChatGPT normal mode: error-analysis narrative
```text
Help me write an error analysis section for an ICU escalation benchmark paper. Use these error categories: missed trigger, delayed trigger, wrong action family, insufficient evidence retrieval, contradiction not caught, unsafe escalation, unnecessary abstention, and failure to abstain. The tone should be critical and specific, not defensive.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/07_results.md` and `paper/sections/08_ablations_and_error_analysis.md` with placeholders linked to exported result tables and error-bucket summaries.

## DONE CRITERIA
- Results section is driven by actual outputs, not vague interpretation

---

# Phase I. Limitations, ethics, and reproducibility

## Objective
Preempt reviewer criticism.

## Limitations to state clearly
- retrospective replay only
- protocol-based labels are simplifications
- observed clinician action is not identical to optimal care
- MIMIC may have missingness and institution-specific practice patterns
- note availability may be inconsistent
- benchmark does not certify clinical deployment

## Ethics / safety points
- de-identified public dataset
- benchmark intended for research evaluation, not bedside use
- abstention and provenance are explicit system features

## Prompt for ChatGPT normal mode: limitations section
```text
Write a limitations and ethical considerations section for a medical AI benchmark paper on event-triggered ICU escalation agents. Be rigorous and explicit about retrospective design, protocol-based labels, data missingness, and why benchmark success does not imply deployment readiness.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/sections/09_limitations_ethics_reproducibility.md` with placeholders for data-use statement, reproducibility artifacts, limitations, and release checklist.

## DONE CRITERIA
- Section is candid and reviewer-proofed

---

# Phase J. Figures and tables plan

## Objective
Plan visuals early so implementation exports what the paper needs.

## Required figures
1. System overview diagram
2. Benchmark replay diagram with one example episode
3. Task definition figure or timeline example
4. Main results figure
5. Ablation figure
6. Error analysis figure

## Required tables
1. Related work comparison
2. MIMIC tables used and their role
3. Task definitions and action windows
4. Benchmark statistics
5. Main results
6. Ablations
7. Error categories or qualitative examples

## FOR ME
- Sketch figures now, even if rough
- Save figure plan in `paper/figure_plan.md`

## FOR CURSOR
**Prompt to Cursor:**
Create placeholder markdown or CSV files for all core tables and generate scripts that will later export benchmark statistics, main results, and ablations in paper-ready form.

## DONE CRITERIA
- Every section has planned supporting figure/table artifacts

---

# Phase K. Citation workflow

## Objective
Do not lose track of citations while coding.

## FOR ME
- Maintain `paper/references.bib`
- Every paragraph draft should note missing citations in brackets if not yet filled

## Prompt for ChatGPT normal mode: citation support
```text
For the following paragraph draft, identify which claims need citations, what type of source would support them best, and suggest likely relevant papers from agentic medical AI, EHR benchmarks, ICU monitoring, or tool-using medical LLM literature.
```

## FOR CURSOR
**Prompt to Cursor:**
Create `paper/citation_todo.md` and a simple script that scans markdown drafts for `[CITE]` markers and summarizes remaining citation gaps.

## DONE CRITERIA
- Citation debt is visible and manageable

---

# Phase L. Writing schedule synchronized with implementation

## Week 1
### FOR ME
- Literature search
- Related-work spreadsheet
- Draft intro skeleton
- Draft benchmark/task-definition outline

### FOR CURSOR
- Generate paper folder skeleton and section templates

## Week 2
### FOR ME
- Write data/labels section draft from audited triggers
- Refine novelty statement

### FOR CURSOR
- Export task spec summaries and dataset inventory tables

## Week 3
### FOR ME
- Draft system/baselines section
- Draft evaluation section

### FOR CURSOR
- Export baseline comparison and tool list tables

## Week 4+
### FOR ME
- Write results and error analysis only after stable outputs exist

### FOR CURSOR
- Export result tables and figures automatically

---

# Ready-to-use paper prompts bundle

## Prompt 1. Full related work search
```text
I am writing a paper on a benchmark for event-triggered ICU escalation agents on MIMIC-IV. Find the closest literature from 2023 to 2026, especially in NeurIPS, ICML, ICLR, AAAI, KDD, CHIL, MIDL, MLHC, Nature Digital Medicine, NEJM AI, JAMIA, and JMIR AI. Focus on EHR agents, ICU monitoring systems, medical tool-using LLMs, workflow automation, and interactive EHR benchmarks. For each paper, tell me exactly why it is or is not close to our work.
```

## Prompt 2. Novelty attack prompt
```text
Attack my novelty claim for this paper as a skeptical NeurIPS reviewer. The project is a MIMIC-IV event-triggered ICU escalation benchmark and tool-using agent. Identify any literature overlaps, weak novelty wording, and claims that sound too broad. Then rewrite the novelty claim to be maximally defensible.
```

## Prompt 3. Label-method attack prompt
```text
Act as a skeptical reviewer and attack the validity of my deterministic label construction for hyperkalemia, hypoglycemia, and sustained hypotension on MIMIC-IV. Tell me exactly what ambiguities, leakage risks, or confounds a reviewer would point out, and suggest fixes.
```

## Prompt 4. Results-writing prompt
```text
Help me write a rigorous results section for a benchmark-first medical AI paper. The systems are rule-based, XGBoost, one-shot LLM, RAG, and a tool-using Gemini agent. I need the writing to focus on workflow-level metrics, evidence retrieval, abstention, and safety, not just predictive performance.
```

## Prompt 5. Reviewer-style criticism prompt
```text
Review my paper outline like a top-tier reviewer in NeurIPS/CHIL/MLHC. Be harsh. Identify what is underspecified, what claims are weak, what baseline is missing, and what experiment is needed to make the work publishable.
```

---

# Final advice for writing

The strongest version of this paper is:
- a clean benchmark contribution
- a careful label-construction contribution
- a disciplined agent baseline
- a strong non-agent comparison
- honest workflow-level evaluation

The weakest version is:
- claiming broad clinical reasoning
- vague novelty
- under-specified labels
- missing ablations
- paper text drifting away from the implemented benchmark
