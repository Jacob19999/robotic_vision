---

description: "Task list for Phase 1 baseline implementation"
---

# Tasks: Phase 1 Baseline Implementation

**Input**: Design documents from `/specs/001-phase1-baseline/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Validation and smoke-test tasks are included because benchmark integrity, schema
checks, and reproducible local execution are required by the Phase 1 plan.

**Organization**: Tasks are grouped by user story to enable independent implementation and
testing of each story.

**Project Constitution Note**: All tasks stay within Phase 1 real-data baseline work. No
Omniverse, Isaac Sim, synthetic-data, or mixed-data retraining tasks are included.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no blocking dependency)
- **[Story]**: Which user story this task belongs to (`[US1]`, `[US2]`, `[US3]`)
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize the repository for a single-project Phase 1 Python pipeline.

- [X] T001 Initialize project metadata and dependency groups in `./pyproject.toml`
- [X] T002 Create the top-level Python package marker in `src/__init__.py`
- [X] T003 [P] Configure local artifact and environment ignores in `./.gitignore`
- [X] T004 [P] Create shared artifact path helpers in `src/utils/paths.py`
- [X] T005 [P] Configure shared pytest fixtures and temporary artifact roots in `tests/conftest.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared models, validators, and CLI wiring that all user stories depend on.

**CRITICAL**: No user story work should begin until this phase is complete.

- [X] T006 Create ontology entity definitions in `src/data/ontology/models.py`
- [X] T007 [P] Create dataset asset and split entity definitions in `src/data/manifests/models.py`
- [X] T008 [P] Create benchmark manifest schema validator in `src/data/manifests/validator.py`
- [X] T009 [P] Create baseline run report schema validator in `src/evaluation/reporting/validator.py`
- [X] T010 Create source registry and label-mapping loader in `src/data/ontology/registry.py`
- [X] T011 Create split-freeze and leakage-check service in `src/data/splits/service.py`
- [X] T012 [P] Create local hardware profiling utilities in `src/utils/hardware.py`
- [X] T013 Create shared CLI command registration in `src/cli/main.py`
- [X] T014 [P] Add schema contract tests for benchmark and run reports in `tests/schema/test_contract_schemas.py`
- [X] T015 [P] Add split-integrity validation tests in `tests/integration/test_split_integrity.py`

**Checkpoint**: Foundation is ready for story-by-story implementation.

---

## Phase 3: User Story 1 - Build a Trusted Real-Data Benchmark (Priority: P1) MVP

**Goal**: Create a curated, normalized benchmark with a fixed ontology and frozen held-out
real test split.

**Independent Test**: Run the benchmark-preparation workflow on sample inputs and verify
that every accepted example is normalized into the shared ontology, assigned to one split,
and written into a valid benchmark manifest with no held-out leakage.

- [X] T016 [P] [US1] Create the COCO ingestion adapter in `src/data/ingestion/coco.py`
- [X] T017 [P] [US1] Create the Open Images ingestion adapter in `src/data/ingestion/open_images.py`
- [X] T018 [P] [US1] Create source metadata loading utilities in `src/data/ingestion/source_registry.py`
- [X] T019 [US1] Implement annotation normalization and class-mapping logic in `src/data/ingestion/normalize.py`
- [X] T020 [US1] Implement benchmark manifest assembly in `src/data/manifests/builder.py`
- [X] T021 [US1] Implement the benchmark preparation CLI command in `src/cli/prepare_benchmark.py`
- [X] T022 [P] [US1] Create benchmark preparation fixtures in `tests/fixtures/phase1_benchmark/manifest_input.json`
- [X] T023 [US1] Add benchmark preparation integration coverage in `tests/integration/test_prepare_benchmark.py`
- [X] T024 [US1] Create the Phase 1 source and ontology configuration template in `config/phase1.yaml`

**Checkpoint**: User Story 1 is complete when the project can build a valid benchmark
manifest with frozen splits from approved real-image sources.

---

## Phase 4: User Story 2 - Compare Reference and Trainable Baselines (Priority: P2)

**Goal**: Execute one zero-shot reference baseline and one compact trainable baseline on the
same benchmark with comparable outputs.

**Independent Test**: Run both baseline paths against the same benchmark manifest and verify
that both produce valid structured run reports with runtime, VRAM, and core quality metrics.

- [X] T025 [P] [US2] Implement the Grounding DINO baseline runner in `src/models/grounding_dino/runner.py`
- [X] T026 [P] [US2] Implement the Florence-2 baseline runner in `src/models/florence2/runner.py`
- [X] T027 [US2] Implement shared metric computation utilities in `src/evaluation/metrics/service.py`
- [X] T028 [P] [US2] Implement baseline artifact export helpers in `src/evaluation/reporting/artifacts.py`
- [X] T029 [US2] Implement structured baseline run report generation in `src/evaluation/reporting/run_report.py`
- [X] T030 [US2] Implement the baseline execution CLI command in `src/cli/run_baseline.py`
- [X] T031 [P] [US2] Create baseline execution fixtures in `tests/fixtures/phase1_reports/sample_run_input.json`
- [X] T032 [US2] Add baseline execution smoke coverage in `tests/integration/test_run_baseline.py`

**Checkpoint**: User Story 2 is complete when both required baseline families can run
against the same manifest and emit comparable reports.

---

## Phase 5: User Story 3 - Decide the Next Phase 1 Experiment (Priority: P3)

**Goal**: Group failures, summarize tradeoffs, and produce the decision artifact that closes
Phase 1.

**Independent Test**: Generate a Phase 1 summary from completed baseline reports and verify
that it names the preferred path, groups failures by taxonomy, and records blocked work.

- [X] T033 [P] [US3] Implement failure-taxonomy grouping logic in `src/evaluation/failure_analysis/grouping.py`
- [X] T034 [P] [US3] Implement failure visualization export utilities in `src/evaluation/failure_analysis/export.py`
- [X] T035 [US3] Implement Phase 1 comparison and decision-summary generation in `src/evaluation/reporting/phase1_summary.py`
- [X] T036 [US3] Implement the Phase 1 summary CLI command in `src/cli/summarize_phase1.py`
- [X] T037 [P] [US3] Create summary-generation fixtures in `tests/fixtures/phase1_reports/sample_summary_inputs.json`
- [X] T038 [US3] Add decision-summary integration coverage in `tests/integration/test_summarize_phase1.py`

**Checkpoint**: User Story 3 is complete when the project can turn completed baseline runs
into a failure-aware Phase 1 decision summary.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final repo-level alignment, documentation, and end-to-end validation.

- [X] T039 [P] Document local setup and execution expectations in `./README.md`
- [X] T040 [P] Align runnable commands and expected outputs in `specs/001-phase1-baseline/quickstart.md`
- [X] T041 Run an end-to-end Phase 1 dry run and record implementation notes in `artifacts/reports/phase1-dry-run.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: no prerequisites
- **Phase 2: Foundational**: depends on Phase 1 and blocks all user stories
- **Phase 3: User Story 1**: depends on Phase 2 and is the MVP
- **Phase 4: User Story 2**: depends on User Story 1 because both baselines require the
  benchmark manifest produced there
- **Phase 5: User Story 3**: depends on User Story 2 because the decision summary requires
  completed baseline reports
- **Phase 6: Polish**: depends on all selected user stories

### User Story Dependencies

- **User Story 1 (P1)**: first independent deliverable after foundational work
- **User Story 2 (P2)**: requires the benchmark manifest and split protections from US1
- **User Story 3 (P3)**: requires completed baseline reports from US2

### Within Each User Story

- fixtures before integration tests
- ingestion or model runners before CLI wiring
- CLI wiring before end-to-end validation
- report generation before summary validation

### Parallel Opportunities

- Setup tasks `T003`-`T005` can run in parallel after `T001`-`T002`
- Foundational tasks `T007`-`T009`, `T012`, `T014`, and `T015` can run in parallel once the
  core package exists
- User Story 1 ingestion adapters `T016`-`T018` can run in parallel
- User Story 2 model runners `T025` and `T026` can run in parallel, and fixtures `T031`
  can be prepared alongside report helpers `T028`
- User Story 3 failure-analysis tasks `T033` and `T034` can run in parallel

---

## Parallel Example: User Story 1

```text
Task: "T016 [US1] Create the COCO ingestion adapter in src/data/ingestion/coco.py"
Task: "T017 [US1] Create the Open Images ingestion adapter in src/data/ingestion/open_images.py"
Task: "T018 [US1] Create source metadata loading utilities in src/data/ingestion/source_registry.py"
```

## Parallel Example: User Story 2

```text
Task: "T025 [US2] Implement the Grounding DINO baseline runner in src/models/grounding_dino/runner.py"
Task: "T026 [US2] Implement the Florence-2 baseline runner in src/models/florence2/runner.py"
Task: "T028 [US2] Implement baseline artifact export helpers in src/evaluation/reporting/artifacts.py"
```

## Parallel Example: User Story 3

```text
Task: "T033 [US3] Implement failure-taxonomy grouping logic in src/evaluation/failure_analysis/grouping.py"
Task: "T034 [US3] Implement failure visualization export utilities in src/evaluation/failure_analysis/export.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate benchmark integrity and manifest generation before moving forward

### Incremental Delivery

1. Deliver the benchmark pipeline first
2. Add the two required baseline families second
3. Add failure analysis and decision reporting third
4. Finish with documentation alignment and a dry run

### Suggested MVP Scope

The MVP is **User Story 1 only**: benchmark curation, normalization, split freeze, and valid
manifest generation. This is the smallest deliverable that creates durable value for the
rest of Phase 1.

---

## Notes

- Total tasks: 41
- User Story 1 tasks: 9
- User Story 2 tasks: 8
- User Story 3 tasks: 6
- All tasks follow the required checklist format with checkbox, task ID, optional parallel
  marker, required story label for story phases, and exact file path
