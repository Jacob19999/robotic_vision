# Feature Specification: Phase 1 Baseline Implementation

**Feature Branch**: `001-phase1-baseline`  
**Created**: 2026-04-08  
**Status**: Draft  
**Input**: User description: "phase 1 baseline implementation"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Build a Trusted Real-Data Benchmark (Priority: P1)

As the project team, we can assemble a curated real-image household-object benchmark with a
shared ontology and frozen splits so that all baseline comparisons are valid.

**Why this priority**: Without a trusted benchmark, every later model comparison becomes
ambiguous and the Phase 1 baseline cannot be defended.

**Independent Test**: This story can be fully tested by reviewing the benchmark manifest and
confirming that every accepted example is mapped to an in-scope class, normalized into the
shared schema, and assigned to exactly one split, including a frozen held-out real test set.

**Acceptance Scenarios**:

1. **Given** approved real-image sources, **When** benchmark preparation is completed,
  **Then** every accepted example is mapped into the shared class ontology and assigned to
   exactly one split.
2. **Given** a source dataset contains inconsistent or out-of-scope labels, **When**
  normalization is performed, **Then** the mapping or exclusion decision is documented
   before the example is included in the benchmark.

---

### User Story 2 - Compare Reference and Trainable Baselines (Priority: P2)

As the project team, we can evaluate a reference baseline and a compact trainable baseline
on the same benchmark so we can compare quality, runtime, and hardware feasibility fairly.

**Why this priority**: Phase 1 must identify a realistic starting model path, not just
collect data.

**Independent Test**: This story can be fully tested by producing two baseline reports on
the same validation and held-out real test data and verifying that they use the same
evaluation protocol and comparable outcome measures.

**Acceptance Scenarios**:

1. **Given** a prepared benchmark, **When** the reference baseline and compact trainable
  baseline are evaluated, **Then** both reports include the same split definitions and the
   same core comparison metrics.
2. **Given** a baseline run cannot complete within the local hardware budget, **When** the
  evaluation package is produced, **Then** the limitation is recorded and the blocked run is
   not represented as a valid comparison result.

---

### User Story 3 - Decide the Next Phase 1 Experiment (Priority: P3)

As the project lead, we can review grouped failure cases and baseline tradeoffs so we can
decide whether to continue with the preferred baseline path or approve an optional
constrained stretch experiment.

**Why this priority**: The project needs a clear evidence-based decision at the end of Phase
1, not just raw metrics.

**Independent Test**: This story can be fully tested by reviewing the final decision summary
and confirming that the recommended next step is traceable to the benchmark results and
failure analysis.

**Acceptance Scenarios**:

1. **Given** completed baseline evaluations, **When** failure analysis is prepared, **Then**
  errors are grouped into the agreed failure taxonomy with representative examples.
2. **Given** the Phase 1 evaluation package, **When** the decision summary is written,
  **Then** it names the preferred baseline path and states whether any optional stretch
   experiment is justified.

### Edge Cases

- A target class has too few usable real examples after normalization to support a reliable
  split.
- Two approved sources use ambiguous labels that could map to different in-scope classes.
- A near-duplicate image appears across candidate training and held-out real test sources.
- A baseline produces detections for categories outside the approved ontology.
- A run completes partially but cannot finish within the local hardware budget.

## Candidate Data Sources

- **Primary fast-start source**: COCO 2017 is the preferred starting source for the first
  extraction pass because it provides a mature official detection benchmark with fixed splits
  and a manageable category set for initial baseline work.
- **Primary scale-up supplement**: Open Images V7 is the preferred supplement when the
  approved ontology needs more breadth than the fast-start source can provide because it
  offers dense box annotations and supports targeted subset extraction.
- **Broader-coverage supplement**: Objects365 is an approved candidate when additional
  household-category breadth is needed, but it is lower priority than the first two sources
  because it has higher download, cleanup, and access overhead.
- **Optional domain-specific supplement**: Vetted indoor or kitchen community datasets may
  be used only when their license, label quality, and ontology mapping are reviewed before
  inclusion.
- **Held-out evaluation source**: In-house real captures are the preferred source for the
  final held-out real evaluation set because they best reflect the target environment.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The Phase 1 baseline feature MUST define a fixed household-object ontology for
all in-scope real-data experiments.
- **FR-002**: The system MUST ingest curated real-image data from approved sources and
normalize accepted examples into one shared annotation schema.
- **FR-003**: The system MUST record source provenance, class mapping, and split assignment
for every accepted example.
- **FR-004**: The system MUST assign each accepted example to exactly one of `train`, `val`,
or `test_real_heldout`.
- **FR-005**: The held-out real test split MUST be frozen before model selection and MUST
NOT be used for training, tuning, or checkpoint selection.
- **FR-006**: Users MUST be able to evaluate one reference baseline and one compact
trainable baseline against the same benchmark definition.
- **FR-007**: The system MUST generate comparable evaluation outputs for each accepted
baseline run, including overall detection quality, class-level quality where available,
runtime, and memory-feasibility measures.
- **FR-008**: The system MUST record reproducibility metadata for each accepted run,
including the dataset version or manifest, split identity, run settings summary, and
outcome summary.
- **FR-009**: The system MUST produce a qualitative failure-analysis artifact that groups
representative errors by failure cause.
- **FR-010**: The system MUST produce a final Phase 1 decision summary that identifies the
preferred baseline path and documents any deferred or blocked experiments.

### Baseline and Evaluation Constraints *(mandatory for this project)*

- **BEC-001**: Feature scope MUST remain within Phase 1 real-data baseline work unless a
constitution amendment explicitly expands scope.
- **BEC-002**: In-scope data sources MUST be curated real-image datasets or curated in-house
  captures that can be mapped into the shared household-object ontology. The default source
  order is COCO 2017 first, Open Images V7 second, Objects365 third, and vetted
  domain-specific supplements only when justified by missing classes.
- **BEC-003**: The held-out real test split MUST be fixed before baseline comparison begins
and MUST remain isolated from training, tuning, and checkpoint selection.
- **BEC-004**: Required outputs MUST include a benchmark manifest, comparable baseline
evaluation reports, runtime and memory-feasibility summaries, and a failure-analysis
artifact.
- **BEC-005**: Any optional stretch-model experiment MUST be explicitly marked optional and
MUST be justified against the declared local hardware budget before it is considered part
of the Phase 1 baseline package.

### Key Entities *(include if feature involves data)*

- **Household Object Class**: An approved target category in the shared Phase 1 ontology,
including its canonical name and any documented source-label mappings.
- **Dataset Asset**: A single accepted real-image example together with its source
provenance, normalized annotations, and split assignment.
- **Split Definition**: The documented policy and recorded assignment rules for `train`,
`val`, and `test_real_heldout`.
- **Baseline Run**: One completed or blocked benchmark attempt with a run summary, metric
outputs, and hardware-feasibility status.
- **Evaluation Report**: A comparison-ready artifact summarizing baseline outcomes,
reproducibility metadata, and decision-relevant findings.
- **Failure Example**: A representative miss, confusion, false positive, or boundary case
grouped under the agreed failure taxonomy.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of accepted in-scope real-image examples are mapped to the approved
ontology and assigned to exactly one split, with no approved overlap between held-out real
test data and training or validation data.
- **SC-002**: The team can produce comparable evaluation reports for at least two baseline
approaches on the same benchmark, with overall detection quality, precision or recall,
class-level quality where available, runtime, and memory-feasibility reported for each.
- **SC-003**: At least one baseline approach completes its required Phase 1 evaluation
workflow on the declared local workstation without requiring additional off-machine
compute.
- **SC-004**: 100% of accepted baseline runs include reproducibility metadata and at least
10 representative failure examples, or all available failure examples if fewer than 10
exist.
- **SC-005**: A final decision summary identifies the preferred Phase 1 baseline path, names
the top failure categories, and explicitly states whether any optional stretch experiment
is justified.

## Assumptions

- The first Phase 1 baseline cycle targets a compact household-object class set suitable for
indoor scenes rather than an open-ended ontology.
- Bounding-box-style detection output is sufficient for the Phase 1 benchmark.
- Public datasets will be curated into focused subsets rather than mirrored in full.
- The initial extraction pass will likely use COCO 2017 for speed, then add class-filtered
  Open Images V7 or Objects365 slices only where the approved ontology is under-covered.
- The local workstation is the primary execution environment for all required baseline work.
- Synthetic data generation, simulation tooling, and mixed-data retraining are deferred to
  later phases and are not part of this feature.
