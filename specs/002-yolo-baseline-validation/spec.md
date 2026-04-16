# Feature Specification: Phase 1 YOLO Baseline Validation

**Feature Branch**: `002-yolo-baseline-validation`  
**Created**: 2026-04-15  
**Status**: Draft  
**Input**: User description: "For the next phase, phase one, let's add an additional model. We'll use one of the latest YOLO models and add it to the existing baseline. Make sure that it supports the existing data ingestion plan. This will be a modification to phase one where we are adding a third model for baseline validation. In addition, this should support future training with synthetic data generation, but the current phase is limited to just adding YOLO to this implementation."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare a Third Baseline Family Fairly (Priority: P1)

As the project team, we can evaluate a YOLO detector baseline on the same benchmark as the
existing Phase 1 baselines so that we can judge whether a compact detector path improves or
complements the current baseline package.

**Why this priority**: The feature only has value if YOLO results are directly comparable to
the existing Grounding DINO and Florence-2 evidence rather than being treated as a separate
experiment.

**Independent Test**: This story can be fully tested by producing a YOLO baseline report
that references the same benchmark manifest, split definitions, and comparison metrics used
by the current baseline reports.

**Acceptance Scenarios**:

1. **Given** a prepared benchmark manifest, **When** the YOLO baseline is evaluated,
   **Then** the resulting report uses the same split identities, class ontology, and core
   comparison metrics as the existing baseline reports.
2. **Given** the YOLO baseline cannot complete within the declared local hardware budget,
   **When** the evaluation package is produced, **Then** the blocked condition is recorded
   without misrepresenting the run as a valid completed comparison.

---

### User Story 2 - Preserve the Existing Ingestion Contract (Priority: P2)

As the data curator, we can add YOLO baseline support without creating a separate ingestion
workflow so that all approved real-image sources continue to flow through one shared
ontology, normalization process, and split policy.

**Why this priority**: A third model is only trustworthy if it reuses the same benchmark
definition instead of introducing an inconsistent data path.

**Independent Test**: This story can be fully tested by confirming that the YOLO baseline
consumes benchmark assets derived from the same normalized manifest and does not require a
second source-specific labeling or split-management process.

**Acceptance Scenarios**:

1. **Given** COCO, Open Images, or in-house captures have already been normalized into the
   benchmark manifest, **When** the YOLO baseline is prepared, **Then** class mappings and
   split assignments are inherited from the authoritative manifest rather than recreated by
   hand.
2. **Given** the benchmark manifest is regenerated from approved sources, **When** the YOLO
   baseline is rerun, **Then** its training and evaluation inputs can be regenerated from
   the same manifest version used by the other baseline families.

---

### User Story 3 - Keep the Detector Path Ready for Later Synthetic Expansion (Priority: P3)

As the project lead, we can add YOLO in a way that stays compatible with future synthetic
training work so that later phases can extend the detector path without redesigning Phase 1
evaluation rules.

**Why this priority**: The current phase is still Phase 1 only, but the added detector
baseline should not force a later rework when synthetic data is introduced.

**Independent Test**: This story can be fully tested by reviewing the feature artifacts and
confirming that YOLO uses the same canonical classes, split protections, and reproducibility
records that future approved synthetic assets would need to follow.

**Acceptance Scenarios**:

1. **Given** future synthetic assets are approved in a later phase, **When** they are mixed
   into detector training data, **Then** they can follow the same canonical ontology and
   split policy without altering the frozen held-out real test protocol.
2. **Given** this Phase 1 feature is reviewed, **When** scope is checked, **Then** the
   artifacts describe only the YOLO baseline addition and explicitly defer synthetic data
   generation and mixed-data retraining.

### Edge Cases

- The YOLO baseline requires a class index order that diverges from the canonical household
  ontology.
- A YOLO-oriented training view drops ignored annotations, source provenance, or split
  assignments that exist in the authoritative benchmark manifest.
- The chosen YOLO variant can run inference locally but cannot complete the required
  training or validation workflow within the workstation memory budget.
- A future synthetic dataset uses a different box convention and must be reconciled without
  weakening the existing held-out real test isolation.
- The YOLO baseline produces metrics or failure artifacts that are not directly comparable
  with the current baseline reporting package.

## Baseline Extension Scope

- **Unchanged reference baseline**: Grounding DINO remains the zero-shot reference path.
- **Unchanged compact trainable baseline**: Florence-2 remains part of the Phase 1 baseline
  package.
- **New detector validation baseline**: A recent compact YOLO-family detector is added as a
  third comparison model for Phase 1 baseline validation.
- **Future-facing compatibility rule**: Any YOLO-specific training or evaluation view must
  be derived from the same benchmark manifest and ontology so that later approved synthetic
  assets can extend the detector path without redefining Phase 1 evaluation boundaries.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST preserve the existing Phase 1 household-object ontology and
  benchmark manifest as the authoritative source of truth for all baseline families.
- **FR-002**: The system MUST support one approved recent YOLO-family detector as an
  additional Phase 1 baseline family alongside the existing baselines.
- **FR-003**: Users MUST be able to compare Grounding DINO, Florence-2, and YOLO baseline
  outcomes using the same split definitions, class mappings, and core comparison metrics.
- **FR-004**: Any YOLO-specific training or evaluation view MUST be derived from already
  normalized benchmark assets rather than a separate ingestion workflow.
- **FR-005**: The system MUST preserve a documented mapping between canonical class
  identifiers and any YOLO-specific class ordering used for baseline execution.
- **FR-006**: The held-out real test split MUST remain excluded from YOLO training, tuning,
  and checkpoint selection exactly as it is excluded for the existing baseline families.
- **FR-007**: The system MUST preserve source provenance, normalized annotations, and split
  assignment metadata when preparing any YOLO-oriented training or evaluation assets.
- **FR-008**: The system MUST generate a comparison-ready baseline report for each YOLO run,
  including completed, blocked, or failed outcomes.
- **FR-009**: The system MUST record reproducibility metadata for each YOLO run, including
  the benchmark manifest identity, split versions, model identity, image-size setting,
  precision mode, batch size, and outcome summary.
- **FR-010**: The feature MUST define a hardware-fit fallback for the YOLO baseline so that
  a smaller approved variant can be used if the initial candidate exceeds the local
  workstation budget.
- **FR-011**: The YOLO baseline path MUST remain compatible with future approved synthetic
  training assets by continuing to use the canonical ontology, shared split policy, and
  held-out real test isolation rules.
- **FR-012**: The Phase 1 decision package MUST be able to incorporate YOLO results as a
  third validation track without representing synthetic generation or mixed-data retraining
  as current-phase deliverables.

### Baseline and Evaluation Constraints *(mandatory for this project)*

- **BEC-001**: Feature scope MUST remain within Phase 1 real-data baseline work unless a
  constitution amendment explicitly expands scope.
- **BEC-002**: In-scope data sources remain the approved real-image datasets and curated
  in-house captures already governed by the shared Phase 1 ingestion plan; the YOLO feature
  MUST NOT introduce a separate source-admission policy.
- **BEC-003**: The held-out real test split MUST remain fixed before YOLO baseline
  comparison begins and MUST stay isolated from YOLO training, tuning, and checkpoint
  selection.
- **BEC-004**: Required outputs MUST include any derived YOLO training-view metadata,
  comparable baseline reports, runtime and memory-feasibility summaries, and failure
  analysis artifacts that can be reviewed beside the existing baselines.
- **BEC-005**: Future synthetic-data support MAY only be expressed as compatibility
  requirements in this feature; synthetic data generation, synthetic ingestion, and mixed
  real-plus-synthetic retraining remain out of scope for the current implementation.

### Key Entities *(include if feature involves data)*

- **Baseline Family**: An approved model path in the Phase 1 comparison set, including its
  role, required outputs, and hardware-feasibility status.
- **Benchmark Manifest**: The authoritative record of accepted assets, canonical classes,
  provenance, and split assignments used by every baseline family.
- **YOLO Class Mapping**: The documented relationship between canonical household-object
  class identifiers and the class ordering expected by the YOLO baseline path.
- **Detector Training View**: A derived, reproducible training or evaluation package built
  from the benchmark manifest for detector-style execution while preserving manifest
  provenance and split protections.
- **Baseline Run**: One completed, blocked, or failed execution attempt against the shared
  benchmark, with comparable metrics and reproducibility metadata.
- **Evaluation Report**: A comparison-ready artifact summarizing quality, runtime, hardware
  fit, and representative failures across the approved baseline families.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The team can produce a YOLO baseline report that references the same benchmark
  manifest and split definitions as the existing baseline reports, with no manual relabeling
  outside the approved ontology.
- **SC-002**: 100% of assets used by the YOLO baseline preserve canonical class mapping,
  source provenance, and split assignment from the authoritative benchmark manifest.
- **SC-003**: At least one approved YOLO baseline variant either completes its required
  Phase 1 workflow on the declared local workstation or produces a blocked report that
  explicitly records the limiting hardware constraint.
- **SC-004**: YOLO baseline results can be included in the same Phase 1 comparison summary
  as the existing baselines, including quality metrics, runtime or memory measurements, and
  representative failure examples.
- **SC-005**: The feature introduces no current-phase requirement for synthetic generation
  or mixed-data retraining and preserves 100% isolation of the held-out real test split.

## Assumptions

- The existing Phase 1 ontology, approved real-data sources, and benchmark manifest remain
  authoritative and are not redefined by this feature.
- A compact recent YOLO-family detector will be chosen during planning based on local
  workstation fit, while larger YOLO variants remain out of scope unless later evidence
  justifies them.
- The YOLO addition is treated as a Phase 1 baseline extension rather than a new project
  phase, because it preserves the real-data-only scope and current evaluation rules.
- Later synthetic-data work will reuse the same canonical class list and held-out real test
  protection, but that work is deferred beyond this feature.
- Existing Phase 1 reporting artifacts can be extended to include a third baseline family
  without changing the core decision criteria.
