# check-implement

This command is the “no surprises” implementation verifier.

When a request references a **phase** or a **part of the implementation**, you must route your work to that exact part of `@IMPLEMENTATION_TASKS.md` and treat it as the **contract**.

## Scope routing (mandatory)
- If the request says “Phase X” or names a section (“Event normalization”, “Replay environment”, etc.), go to that section in `@IMPLEMENTATION_TASKS.md` and treat it as the only scope.
- If the request is ambiguous, resolve it by locating the most specific matching section(s) in `@IMPLEMENTATION_TASKS.md` and explicitly stating which section(s) you are verifying.
- Do not broaden scope beyond v1 invariants (hyperkalemia, hypoglycemia, sustained hypotension).

## Step 1 — Extract the required functionality checklist (mandatory)
From the chosen section(s) of `@IMPLEMENTATION_TASKS.md`, produce an explicit checklist of **everything that must be true**, including:
- required public functions/APIs/CLI behavior
- required schemas and output fields (including “JSON only” constraints when specified)
- required determinism/reproducibility constraints (seeds, manifests, fixed outputs)
- required “no future leakage” / timestamp-boundary constraints (where relevant)
- required file outputs and their locations (generated reports, manifests, etc.)
- required tests and what each test must prove (as stated in DONE CRITERIA / prompts)

This checklist must be complete and unambiguous. If the doc is vague, tighten it into testable statements *without changing intent*.

## Step 2 — Map checklist items to concrete verification methods (mandatory)
For each checklist item, specify at least one verification method:
- existing test(s) that cover it (you may reuse these, but they are **never sufficient by themselves** for this command), or
- a new **throwaway** test that covers it, or
- an end-to-end CLI run / public-interface run that asserts outputs + invariants.

In addition to mapping items individually, you **must plan for at least one single temporary test file** that exercises the **entire contract together** for the scoped section (end-to-end through public interfaces). You are not allowed to conclude that implementation is adequate based only on scattered existing tests.

Avoid “unit-test-only” verification when the contract is end-to-end. Prefer testing through the **public interfaces** used by the benchmark (CLI, data access layer entrypoints, normalizer APIs, etc.).

## Step 3 — Run existing tests first (mandatory)
Run the full existing test suite (e.g., `pytest`) and make it pass. Do not proceed to “looks good” claims until the suite is green.

If any tests are skipped, you must classify each skip as either:
- **Expected (CI-only)**: licensed/local data is truly absent, or
- **Unexpected**: data exists but configuration/detection caused a skip.

When a repo-local MIMIC mirror exists (for example `physionet.org/files/mimiciv/<version>/`), any “real-data” integration test must **not** skip; fix configuration and/or the test to auto-detect and run.

## Step 4 — Create throwaway real-data tests (mandatory)
Create additional **temporary test files** designed specifically to validate the checklist items that are not already strongly covered.

You **must** include at least **one dedicated temporary test file** that:
- lives in a single file (even if it calls helpers),
- exercises the scoped section’s behavior **end-to-end** through public interfaces (CLI or primary APIs),
- asserts something that can only pass if the key pieces of that section are working **together**, not just in isolation.

Hard requirements for these temporary tests:
- They must exercise code through **public interfaces end-to-end** whenever possible.
- They must prefer **real MIMIC-derived data already present in the repository** (or small, well-scoped slices of it) over synthetic fixtures.
- They must validate correctness via assertions on:
  - schema/field presence and types
  - timestamp boundary behavior (no future leakage) when applicable
  - deterministic/reproducible outputs when applicable
  - mapping/normalization coverage expectations (including “flag ambiguous/unmapped” behavior)
- They must be safe and fast: keep data slices small, keep runtime reasonable.

If the repo does not contain any real MIMIC-derived slices usable for the target feature, fall back to the smallest fixture that still tests the invariant, and clearly state what could not be validated on real data (and why).

## Step 5 — Run and compare: existing tests + temporary tests (mandatory)
- Run the existing suite.
- Run the temporary tests.
- Compare outcomes and investigate any discrepancies (including flaky behavior, nondeterminism, or environment-dependent results).
- If a temporary test reveals a bug, fix the implementation and rerun everything until consistently green.

## Step 6 — Cleanup (mandatory, non-negotiable)
- Remove all temporary test files you created for verification.
- Remove any temporary data/artifacts generated only for those tests (cache dirs, scratch outputs, etc.).
- Ensure the repo returns to a clean state except for intentional implementation changes.

## Final report (mandatory)
Produce a short bulleted report that:
- names the exact `@IMPLEMENTATION_TASKS.md` section(s) verified
- lists the extracted functionality checklist
- states which checklist items were covered by existing tests vs temporary tests vs end-to-end runs
- confirms real-data coverage (what real slices were used) or clearly states gaps
- states the exact commands run (tests/CLI) and the final pass status
