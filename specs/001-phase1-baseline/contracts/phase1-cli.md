# Contract: Phase 1 CLI Surface

## Purpose

Define the expected user-facing CLI commands for the Phase 1 benchmark pipeline.

## Command 1: `prepare_benchmark`

### Responsibility

- ingest approved real-image sources
- normalize labels into the shared ontology
- assign and freeze splits
- emit the benchmark manifest

### Inputs

- `--config`: path to the Phase 1 configuration file
- `--output`: path to the benchmark manifest output file

### Required behavior

- reject assets with unmapped or ambiguous class mappings unless the config explicitly
  defines an exclusion rule
- fail if the held-out real test split would overlap with training or validation assets
- record source provenance and split counts in the output manifest

### Outputs

- benchmark manifest file matching `benchmark-manifest.schema.json`
- non-zero exit code when split integrity or ontology validation fails

## Command 2: `run_baseline`

### Responsibility

- execute one baseline run against a benchmark manifest
- capture runtime, memory-feasibility, and quality metrics
- save failure examples and a structured run report

### Inputs

- `--model`: baseline identifier such as `grounding_dino` or `florence2`
- `--manifest`: path to the benchmark manifest file
- `--report`: path to the structured run report output file

### Required behavior

- reject runs that point to an invalid or missing benchmark manifest
- record the baseline mode (`zero_shot`, `trainable`, or `optional_stretch`)
- emit a structured report even when the run ends as `blocked` or `failed`

### Outputs

- baseline report file matching `baseline-run-report.schema.json`
- failure-example artifact directory or documented empty result

## Command 3: `summarize_phase1`

### Responsibility

- compare completed baseline reports
- summarize top failure categories
- generate the Phase 1 decision summary

### Inputs

- `--reports`: directory or glob containing baseline report files
- `--output`: path to the summary output file

### Required behavior

- include only reports generated from the same benchmark manifest in a single comparison
- identify blocked or deferred experiments explicitly
- reject summary generation if no completed required baseline report is available

### Outputs

- structured Phase 1 summary artifact
- non-zero exit code when report comparability rules are violated
