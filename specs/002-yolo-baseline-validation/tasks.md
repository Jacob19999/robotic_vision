---

description: "Task list for Phase 1 YOLO baseline validation"

---

# Tasks: Phase 1 YOLO Baseline Validation

**Input**: Design documents from `/specs/002-yolo-baseline-validation/`  
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Validation, conversion, and runner smoke-test tasks are included because the
feature specification and implementation plan explicitly require schema validation,
detector-view conversion tests, runner smoke tests, and reproducible Phase 1 reporting.

**Organization**: Tasks are grouped by user story to enable independent implementation and
testing of each story.

**Project Constitution Note**: All tasks remain within Phase 1 real-data baseline work.
Tasks preserve dataset normalization, split protection, comparable baseline reporting, and
failure-analysis outputs. No Omniverse, Isaac Sim, synthetic-data generation, or mixed-data
retraining tasks are included.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel when the task touches different files and has no unmet
  dependency
- **[Story]**: Which user story the task belongs to (`[US1]`, `[US2]`, `[US3]`)
- Every task includes an exact file path

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Prepare the repository for the YOLO detector baseline extension.

- [X] T001 Add the Ultralytics dependency for the YOLO baseline workflow in `pyproject.toml`
- [X] T002 [P] Create the detector-view package marker in `src/data/views/__init__.py`
- [X] T003 [P] Create the YOLO11 baseline package marker in `src/models/yolo11/__init__.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared contract resolution, schema models, and report primitives required
before any YOLO story work begins.

**CRITICAL**: No user story work should begin until this phase is complete.

- [X] T004 Resolve the active feature contracts from branch or environment selection in `src/utils/paths.py`
- [X] T005 [P] Define `YOLOClassMapping` and `DetectorDatasetView` models in `src/data/views/models.py`
- [X] T006 [P] Add detector-view schema validation helpers in `src/data/views/validator.py`
- [X] T007 [P] Extend shared runner result primitives with execution-config fields in `src/models/common/predictions.py`
- [X] T008 Update shared run-report construction to require execution configuration for every baseline family in `src/evaluation/reporting/run_report.py`
- [X] T009 [P] Pin test sessions to the `002-yolo-baseline-validation` contracts in `tests/conftest.py`
- [X] T010 [P] Extend shared schema contract coverage for detector views and `execution_config` in `tests/schema/test_contract_schemas.py`

**Checkpoint**: Feature-aware contracts, detector-view schemas, and report primitives are in
place for story-by-story implementation.

---

## Phase 3: User Story 1 - Compare a Third Baseline Family Fairly (Priority: P1) 🎯 MVP

**Goal**: Run YOLO11 beside Grounding DINO and Florence-2 with comparable reporting and
blocked-run handling.

**Independent Test**: Using a valid benchmark manifest and detector-view metadata fixture,
run `phase1 run-baseline --model yolo11` and `phase1 summarize-phase1` to verify that YOLO
produces either a comparable completed report or a structured blocked report alongside the
existing baseline families.

### Tests for User Story 1

- [X] T011 [P] [US1] Add YOLO detector-view fixture data in `tests/fixtures/phase1_yolo/sample_detector_view.json`
- [X] T012 [P] [US1] Update the three-family summary fixtures with YOLO coverage in `tests/fixtures/phase1_reports/sample_summary_inputs.json`
- [X] T013 [P] [US1] Add YOLO11 runner smoke coverage with stub Ultralytics results in `tests/unit/test_yolo11_runner.py`
- [X] T014 [P] [US1] Extend baseline execution integration coverage for `--model yolo11` and `--dataset-view` in `tests/integration/test_run_baseline.py`

### Implementation for User Story 1

- [X] T015 [US1] Implement the Ultralytics YOLO11 baseline runner with `yolo11s` primary and `yolo11n` fallback variants in `src/models/yolo11/runner.py`
- [X] T016 [US1] Extend unified baseline execution for YOLO dataset-view runs in `src/cli/run_baseline.py`
- [X] T017 [US1] Keep Phase 1 comparison summaries fair across Grounding DINO, Florence-2, and YOLO11 in `src/evaluation/reporting/phase1_summary.py`

**Checkpoint**: User Story 1 is complete when YOLO11 can be compared beside the existing
baseline families without changing the shared report semantics.

---

## Phase 4: User Story 2 - Preserve the Existing Ingestion Contract (Priority: P2)

**Goal**: Derive a reproducible YOLO dataset view from the authoritative benchmark manifest
instead of introducing a second ingestion workflow.

**Independent Test**: Run `phase1 export-yolo-view` on a benchmark manifest and verify that
the exported dataset YAML, label files, and metadata preserve canonical class mappings,
split assignments, and manifest counts.

### Tests for User Story 2

- [X] T018 [P] [US2] Add benchmark-manifest export fixtures for detector-view generation in `tests/fixtures/phase1_yolo/sample_benchmark_manifest.json`
- [X] T019 [P] [US2] Add YOLO label conversion coverage for normalized `xywh` output in `tests/unit/test_yolo_view_export.py`
- [X] T020 [P] [US2] Add detector-view export CLI integration coverage in `tests/integration/test_export_yolo_view.py`

### Implementation for User Story 2

- [X] T021 [US2] Implement canonical class-order mapping and normalized `xywh` conversion helpers in `src/data/views/yolo.py`
- [X] T022 [US2] Implement detector-view dataset export and metadata assembly in `src/data/views/exporter.py`
- [X] T023 [US2] Implement the `export-yolo-view` CLI command in `src/cli/export_yolo_view.py`
- [X] T024 [US2] Register `export-yolo-view` in the shared CLI in `src/cli/main.py`

**Checkpoint**: User Story 2 is complete when detector-ready YOLO assets can be regenerated
from the benchmark manifest without any parallel ingestion path.

---

## Phase 5: User Story 3 - Keep the Detector Path Ready for Later Synthetic Expansion (Priority: P3)

**Goal**: Preserve reproducibility, split isolation, and canonical mapping rules so later
approved detector extensions can build on the Phase 1 YOLO path without redesign.

**Independent Test**: Review a generated detector view plus a completed or blocked YOLO run
report and verify that manifest provenance, held-out split isolation, class ordering, and
execution configuration are all explicit and machine-validatable.

### Tests for User Story 3

- [X] T025 [P] [US3] Add detector-view provenance and split-isolation coverage in `tests/unit/test_detector_view_metadata.py`
- [X] T026 [P] [US3] Extend baseline execution coverage for manifest-view mismatch and blocked hardware outcomes in `tests/integration/test_run_baseline.py`
- [X] T027 [P] [US3] Add execution-config and hardware-profile regression coverage in `tests/unit/test_run_report.py`

### Implementation for User Story 3

- [X] T028 [US3] Enforce manifest-view provenance matching and held-out split protection in `src/cli/run_baseline.py`
- [X] T029 [US3] Preserve source provenance, split exports, and future-safe class mapping in `src/data/views/exporter.py`
- [X] T030 [US3] Record fallback model selection and blocked-run notes for YOLO11 in `src/models/yolo11/runner.py`
- [X] T031 [US3] Harden hardware profile collection for schema-compliant reports in `src/utils/hardware.py`
- [X] T032 [US3] Extend Phase 1 settings with YOLO baseline defaults without expanding current scope in `src/config/phase1_settings.py`
- [X] T033 [US3] Document YOLO baseline defaults and deferred synthetic compatibility in `config/phase1.yaml`

**Checkpoint**: User Story 3 is complete when the detector path records the metadata and
guardrails needed for later approved extensions without adding synthetic work now.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Align documentation and capture the end-to-end implementation outcome.

- [X] T034 [P] Update the detector-baseline workflow documentation in `README.md`
- [X] T035 [P] Align the feature quickstart with `export-yolo-view` and YOLO11 execution in `specs/002-yolo-baseline-validation/quickstart.md`
- [X] T036 Run an end-to-end YOLO baseline dry run and record implementation notes in `artifacts/reports/phase1-yolo-dry-run.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1: Setup**: no prerequisites
- **Phase 2: Foundational**: depends on Phase 1 and blocks all story work
- **Phase 3: User Story 1**: depends on Phase 2 and delivers the MVP comparison path using
  validated detector-view inputs
- **Phase 4: User Story 2**: depends on Phase 2 and can proceed after the foundation is in
  place, replacing fixture-based detector views with reproducible exports from the benchmark
  manifest
- **Phase 5: User Story 3**: depends on User Stories 1 and 2 because it hardens the
  combined runner and export path for future-safe reproducibility
- **Phase 6: Polish**: depends on all selected story work being complete

### User Story Dependencies

- **User Story 1 (P1)**: first deliverable after foundational work; proves fair YOLO
  comparison with the existing baselines
- **User Story 2 (P2)**: depends on foundational detector-view contracts but is otherwise
  independently testable from User Story 1
- **User Story 3 (P3)**: depends on the combined export and run flows from User Stories 1
  and 2

### Within Each User Story

- Test fixtures and regression tests before implementation
- Shared data models and validators before CLI wiring
- CLI wiring before end-to-end validation
- Reproducibility and split-protection hardening before documentation updates

### Parallel Opportunities

- Setup tasks `T002` and `T003` can run in parallel after `T001`
- Foundational tasks `T005`-`T007`, `T009`, and `T010` can run in parallel after `T004`
- User Story 1 test tasks `T011`-`T014` can run in parallel
- User Story 2 test tasks `T018`-`T020` can run in parallel
- User Story 3 test tasks `T025`-`T027` can run in parallel
- Polish tasks `T034` and `T035` can run in parallel after implementation stabilizes

---

## Parallel Example: User Story 1

```text
Task: "T011 [US1] Add YOLO detector-view fixture data in tests/fixtures/phase1_yolo/sample_detector_view.json"
Task: "T012 [US1] Update the three-family summary fixtures with YOLO coverage in tests/fixtures/phase1_reports/sample_summary_inputs.json"
Task: "T013 [US1] Add YOLO11 runner smoke coverage with stub Ultralytics results in tests/unit/test_yolo11_runner.py"
Task: "T014 [US1] Extend baseline execution integration coverage for --model yolo11 and --dataset-view in tests/integration/test_run_baseline.py"
```

## Parallel Example: User Story 2

```text
Task: "T018 [US2] Add benchmark-manifest export fixtures for detector-view generation in tests/fixtures/phase1_yolo/sample_benchmark_manifest.json"
Task: "T019 [US2] Add YOLO label conversion coverage for normalized xywh output in tests/unit/test_yolo_view_export.py"
Task: "T020 [US2] Add detector-view export CLI integration coverage in tests/integration/test_export_yolo_view.py"
```

## Parallel Example: User Story 3

```text
Task: "T025 [US3] Add detector-view provenance and split-isolation coverage in tests/unit/test_detector_view_metadata.py"
Task: "T026 [US3] Extend baseline execution coverage for manifest-view mismatch and blocked hardware outcomes in tests/integration/test_run_baseline.py"
Task: "T027 [US3] Add execution-config and hardware-profile regression coverage in tests/unit/test_run_report.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. Validate YOLO comparison behavior with fixture-backed detector-view metadata before
   building the export path

### Incremental Delivery

1. Deliver feature-aware contracts, detector-view models, and report primitives first
2. Add YOLO11 baseline execution and comparison reporting second
3. Add reproducible detector-view export from the benchmark manifest third
4. Harden provenance, split isolation, and fallback metadata fourth
5. Finish with documentation and a dry run

### Suggested MVP Scope

The MVP is **User Story 1 only**: prove that YOLO11 can participate in the Phase 1
comparison workflow with the same report contract as Grounding DINO and Florence-2. User
Story 2 then makes the detector view reproducible from the benchmark manifest, and User
Story 3 hardens the path for future approved extensions.

---

## Notes

- Total tasks: 36
- User Story 1 tasks: 7
- User Story 2 tasks: 7
- User Story 3 tasks: 9
- Parallel task groups identified in every story phase plus setup, foundational, and polish
- All tasks follow the required checklist format with checkbox, task ID, optional parallel
  marker, required story label for story phases, and exact file path
