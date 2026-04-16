# Implementation Plan: Phase 1 YOLO Baseline Validation

**Branch**: `002-yolo-baseline-validation` | **Date**: 2026-04-15 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-yolo-baseline-validation/spec.md`

## Summary

Extend the existing Phase 1 baseline pipeline with a detector-oriented validation track built
around Ultralytics YOLO11. The benchmark manifest remains the authoritative real-data
record; a new export step derives a reproducible YOLO-format dataset view from that
manifest, and a new YOLO11 runner feeds comparable reporting into the same Phase 1 summary
workflow as Grounding DINO and Florence-2.

## Technical Context

**Language/Version**: Python 3.11 target via uv-managed project environment  
**Primary Dependencies**: PyTorch, torchvision, Transformers, Ultralytics, datasets, pycocotools, pandas, Pydantic, Typer, Pillow, PyYAML, jsonschema  
**Storage**: Local filesystem artifacts (images, JSON manifests, YOLO label text files, dataset YAML, checkpoints, failure galleries, reports)  
**Testing**: pytest, schema validation checks, detector-view conversion tests, runner smoke tests with stub predictors/trainers  
**Target Platform**: Windows workstation with NVIDIA RTX 5070 12 GB VRAM, AMD Ryzen 9 9900X, 32 GB RAM  
**Project Type**: Single-project ML and data pipeline with CLI utilities  
**Performance Goals**: Export a deterministic YOLO dataset view from the shared benchmark, run Grounding DINO and Florence-2 plus at least one YOLO11 detector baseline end-to-end locally, and record latency and VRAM for every accepted run  
**Constraints**: Phase 1 only, real-data only, frozen held-out real test split, benchmark manifest remains the source of truth, YOLO labels must be derived in normalized `xywh`, single-GPU local execution, compact YOLO variants only  
**Scale/Scope**: 10-15 household classes, COCO 2017 starter subset, targeted Open Images V7 supplement, 3 comparable Phase 1 baseline families, 1 detector-view export contract

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Phase 1 only`: PASS. The plan adds YOLO detector export and evaluation to the real-data
  baseline package and explicitly defers synthetic data generation, mixed-data retraining,
  Omniverse, and Isaac Sim implementation.
- `Hardware fit`: PASS. The plan targets YOLO11 detect with `yolo11s` as the primary
  variant, `yolo11n` as the fallback, 640-pixel default image size, mixed precision, and
  small-batch local execution sized for the RTX 5070 12 GB VRAM budget.
- `Dataset integrity`: PASS. The benchmark manifest remains authoritative, the derived YOLO
  view reuses the shared ontology and split assignments, and `test_real_heldout` remains
  evaluation-only.
- `Reproducibility`: PASS. The plan adds detector-view metadata, stable class-order
  mapping, execution configuration in run reports, and the same failure-analysis outputs
  required for the existing baselines.
- `Decision gates`: PASS. The execution order is benchmark preparation -> Grounding DINO
  reference -> Florence-2 trainable baseline -> YOLO11 detector validation -> failure
  review -> optional stretch experiment only if justified.

Post-design re-check: PASS. Research, data model, contracts, quickstart, and agent context
stay within the Phase 1 constitution and do not introduce current-phase synthetic or
stretch-model work.

## Project Structure

### Documentation (this feature)

```text
specs/002-yolo-baseline-validation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/
│   └── requirements.md
├── contracts/
│   ├── phase1-cli.md
│   ├── benchmark-manifest.schema.json
│   ├── baseline-run-report.schema.json
│   └── detector-dataset-view.schema.json
└── tasks.md
```

### Source Code (repository root)

```text
src/
├── cli/
├── config/
├── data/
│   ├── ingestion/
│   ├── manifests/
│   ├── ontology/
│   ├── splits/
│   └── views/
├── evaluation/
│   ├── failure_analysis/
│   ├── metrics/
│   └── reporting/
├── models/
│   ├── common/
│   ├── florence2/
│   ├── grounding_dino/
│   └── yolo11/
└── utils/

tests/
├── fixtures/
├── integration/
├── schema/
└── unit/

artifacts/
├── datasets/
├── failure_examples/
├── manifests/
└── reports/
```

**Structure Decision**: Use the existing single-project pipeline and add one derived
detector-view export layer plus one model family module, rather than creating a second
ingestion flow or a separate training project.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | The current plan passes constitution gates without exceptions |
