# Contract: Phase 1 CLI Surface

## Purpose

Define the expected user-facing CLI commands for the Phase 1 benchmark pipeline after the
YOLO baseline extension.

## Command 1: `prepare-benchmark`

### Responsibility

- ingest approved real-image sources
- normalize labels into the shared ontology
- assign and freeze splits
- emit the authoritative benchmark manifest

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

## Command 2: `export-yolo-view`

### Responsibility

- derive a detector-oriented dataset view from an authoritative benchmark manifest
- convert canonical annotations into Ultralytics YOLO label files
- emit detector-view metadata used by the YOLO baseline runner

### Inputs

- `--manifest`: path to the benchmark manifest file
- `--output-dir`: directory where dataset YAML, image links or copies, and label files are
  written
- `--metadata`: path to the detector-view metadata JSON file

### Required behavior

- preserve canonical class membership, split assignment, and source provenance from the
  benchmark manifest
- emit one stable class-order mapping for every exported split
- convert bounding boxes from manifest `xyxy` format into normalized YOLO `xywh` format
- export `train`, `val`, and `test_real_heldout` views without reclassifying the held-out
  split as training data

### Outputs

- detector-view metadata matching `detector-dataset-view.schema.json`
- dataset YAML plus label files under the requested output directory

## Command 3: `run-baseline`

### Responsibility

- execute one baseline run against a benchmark manifest
- capture runtime, memory-feasibility, and quality metrics
- save failure examples and a structured run report

### Inputs

- `--model`: baseline identifier such as `grounding_dino`, `florence2`, or `yolo11`
- `--manifest`: path to the benchmark manifest file
- `--report`: path to the structured run report output file
- `--dataset-view`: detector-view metadata path; required when `--model yolo11` is selected

### Required behavior

- reject runs that point to an invalid or missing benchmark manifest
- reject `yolo11` runs that omit or mismatch the required detector-view metadata
- record the baseline mode (`zero_shot`, `trainable`, or `optional_stretch`)
- record execution configuration, including model variant, image size, precision mode, batch
  size, and detector-view linkage when applicable
- emit a structured report even when the run ends as `blocked` or `failed`

### Outputs

- baseline report file matching `baseline-run-report.schema.json`
- failure-example artifact directory or documented empty result

## Command 4: `summarize-phase1`

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
- compare YOLO11 results beside Grounding DINO and Florence-2 using the same summary rules
- reject summary generation if no completed required baseline report is available

### Outputs

- structured Phase 1 summary artifact
- non-zero exit code when report comparability rules are violated
