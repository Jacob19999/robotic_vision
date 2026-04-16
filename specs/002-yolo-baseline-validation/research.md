# Phase 0 Research: Phase 1 YOLO Baseline Validation

## Decision 1: Use Ultralytics YOLO11 detect models with `yolo11s` primary and `yolo11n` fallback

- **Decision**: Implement the new detector baseline on Ultralytics YOLO11 detection models,
  using `yolo11s.pt` as the default training/evaluation variant and `yolo11n.pt` as the
  fallback when the local hardware budget is exceeded.
- **Rationale**: Ultralytics documents YOLO26 as the latest model, but recommends both
  YOLO26 and YOLO11 for stable production workloads. YOLO11 also publishes clear detection
  support across inference, validation, training, and export, with compact `n` and `s`
  variants that fit the Phase 1 workstation constraints more realistically than `m`, `l`,
  or `x`.
- **Alternatives considered**:
  - Use YOLO26 as the default baseline: rejected because it is newer and not necessary for
    the first detector-validation pass when the goal is a stable comparable baseline.
  - Use `yolo11n.pt` as the primary default: rejected because it leaves too much accuracy on
    the table if `yolo11s.pt` already fits the workstation budget.
  - Use `yolo11m.pt` or larger as the first detector baseline: rejected because the extra
    memory and compute cost are unnecessary for the initial Phase 1 validation track.

## Decision 2: Derive a YOLO dataset view from the benchmark manifest instead of adding a new ingestion path

- **Decision**: Export a detector-specific dataset view from the existing benchmark manifest
  using the Ultralytics YOLO dataset format: dataset YAML plus one normalized label file per
  image in `class x_center y_center width height` form.
- **Rationale**: Ultralytics documents a straightforward dataset YAML structure with
  train/validation/testing paths and normalized `xywh` text labels. Converting from the
  canonical benchmark manifest into that view preserves one source of truth for ontology,
  provenance, and split assignments while satisfying the detector training interface.
- **Alternatives considered**:
  - Add YOLO-specific ingestion configuration alongside the benchmark manifest: rejected
    because it would create a second source-of-truth path for classes and splits.
  - Convert labels on the fly inside the YOLO runner only: rejected because it hides a
    reproducibility-critical artifact and makes schema validation harder.
  - Use Ultralytics NDJSON as the first export format: rejected because the simpler YAML plus
    label-text format is better aligned with current Phase 1 needs and existing expectations.

## Decision 3: Add a dedicated `export-yolo-view` CLI command between benchmark preparation and YOLO execution

- **Decision**: Extend the CLI surface with a dedicated `export-yolo-view` command that reads
  the benchmark manifest and writes a detector dataset view directory plus a metadata JSON
  contract consumed by `run-baseline --model yolo11`.
- **Rationale**: A dedicated export step makes the manifest-to-detector transformation
  explicit, testable, and reproducible. It also keeps `prepare-benchmark` model-agnostic and
  allows future synthetic assets to be incorporated through the same export contract once a
  later phase approves them.
- **Alternatives considered**:
  - Extend `prepare-benchmark` to emit YOLO files automatically: rejected because it would
    tie benchmark curation to one model family and blur the authoritative-vs-derived
    artifact boundary.
  - Hide export inside `run-baseline`: rejected because it makes debugging and regeneration
    harder and weakens the reproducibility trail.

## Decision 4: Keep `run-baseline` as the single execution entrypoint and require detector-view metadata for YOLO runs

- **Decision**: Preserve `run-baseline` as the unified execution command, extend it to accept
  `yolo11` as a model family, and require a detector-view metadata path when the selected
  model family is YOLO11.
- **Rationale**: One execution entrypoint keeps report generation and summary comparison
  centralized. Requiring the detector-view metadata for YOLO runs ensures the train/val/test
  split usage and class mapping remain explicit and reproducible.
- **Alternatives considered**:
  - Create a separate YOLO-only run command: rejected because it would duplicate the
    reporting flow and comparison logic already present in `run-baseline`.
  - Treat YOLO as inference-only: rejected because the feature is specifically about adding a
    trainable detector validation baseline that can later extend to synthetic-data training.

## Decision 5: Extend run-report contracts with explicit execution configuration and activate feature-local contracts during implementation

- **Decision**: Add an `execution_config` object to the baseline run-report schema and plan
  the runtime code to resolve contracts from the active feature directory instead of
  remaining pinned to `001-phase1-baseline`.
- **Rationale**: The current run-report schema does not formally capture model variant,
  image size, precision mode, batch size, or detector-view linkage even though those are
  required for reproducibility. The current contract-resolution helper also defaults to the
  older feature directory, which would cause the YOLO-specific contract updates to be
  ignored unless the runtime becomes feature-aware.
- **Alternatives considered**:
  - Store execution details in free-form `notes`: rejected because it is not machine
    verifiable and would weaken comparison quality.
  - Add YOLO-only top-level report fields without a shared execution-config object: rejected
    because all baseline families benefit from the same reproducibility shape.
  - Leave contract resolution pinned to `001-phase1-baseline`: rejected because the feature's
    new schemas would not become active for the implementation and test surface.
