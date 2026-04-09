# Implementation Plan: Phase 1 Baseline Implementation

**Branch**: `001-phase1-baseline` | **Date**: 2026-04-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-phase1-baseline/spec.md`

## Summary

Implement the Phase 1 real-data baseline pipeline for household-object detection. The plan
creates a curated benchmark with a fixed ontology and frozen held-out real test split, then
evaluates a required zero-shot reference baseline and a required compact trainable baseline
under the local workstation constraints. The output of this phase is a reproducible
benchmark manifest, comparable evaluation reports, failure-analysis artifacts, and a written
decision on the preferred baseline path.

## Technical Context

**Language/Version**: Python 3.11 target via uv-managed project environment  
**Primary Dependencies**: PyTorch, torchvision, Transformers, datasets, pycocotools, pandas, Typer, Pillow  
**Storage**: Local filesystem artifacts (images, JSON manifests, CSV/Parquet summaries, report folders)  
**Testing**: pytest, schema validation checks, dataset split integrity checks, smoke baseline runs  
**Target Platform**: Windows workstation with NVIDIA RTX 5070 12 GB VRAM, AMD Ryzen 9 9900X, 32 GB RAM  
**Project Type**: Single-project ML and data pipeline with CLI utilities  
**Performance Goals**: Prepare the benchmark locally, run Grounding DINO evaluation and at least one Florence-2 baseline end-to-end locally, and record latency and VRAM for every accepted run  
**Constraints**: Phase 1 only, real-data only, frozen held-out real test split, single-GPU local execution, curated subsets instead of full dataset mirrors, optional stretch experiments only after required gates pass  
**Scale/Scope**: 10-15 household classes, COCO 2017 starter subset, targeted Open Images V7 supplement, optional Objects365 backfill, 2 required baseline families plus 1 optional gated stretch family

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- `Phase 1 only`: PASS. This plan is limited to benchmark curation, baseline evaluation, and
  failure analysis on real-image data. Omniverse, Isaac Sim, synthetic data, and mixed-data
  retraining are deferred.
- `Hardware fit`: PASS. Required work is scoped to Grounding DINO and Florence-2 on curated
  subsets with memory-aware settings sized for the RTX 5070 12 GB VRAM budget.
- `Dataset integrity`: PASS. The plan requires one shared ontology, explicit source mapping,
  frozen `test_real_heldout`, provenance tracking, and split-leakage checks.
- `Reproducibility`: PASS. The plan includes versioned benchmark manifests, run reports,
  metric capture, and failure-analysis artifacts as required outputs.
- `Decision gates`: PASS. The execution order is benchmark preparation -> zero-shot baseline
  -> compact trainable baseline -> failure review -> optional stretch experiment only if
  justified.

Post-design re-check: PASS. Research, data model, contracts, and quickstart remain within
Phase 1 scope and preserve the constitution gates above.

## Project Structure

### Documentation (this feature)

```text
specs/001-phase1-baseline/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── checklists/
│   └── requirements.md
└── contracts/
    ├── phase1-cli.md
    ├── benchmark-manifest.schema.json
    └── baseline-run-report.schema.json
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
│   └── splits/
├── evaluation/
│   ├── failure_analysis/
│   ├── metrics/
│   └── reporting/
├── models/
│   ├── common/
│   ├── florence2/
│   └── grounding_dino/
└── utils/

tests/
├── fixtures/
├── integration/
├── schema/
└── unit/

artifacts/
├── failure_examples/
├── manifests/
└── reports/
```

**Structure Decision**: Use a single Python project because Phase 1 is a tightly coupled
benchmarking pipeline with shared data ingestion, evaluation, and reporting logic rather
than separate deployable services.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | The current plan passes constitution gates without exceptions |
