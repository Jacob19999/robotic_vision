# Data Model: Phase 1 YOLO Baseline Validation

## Entity: BaselineFamily

- **Purpose**: Canonical description of one approved baseline path in the Phase 1 comparison
  set.
- **Fields**:
  - `model_family`: Stable family identifier such as `grounding_dino`, `florence2`, or
    `yolo11`
  - `default_variant`: Default executable variant for the family
  - `fallback_variant`: Smaller approved fallback variant when hardware limits are hit
  - `run_mode`: `zero_shot`, `trainable`, or `optional_stretch`
  - `requires_dataset_view`: Boolean indicating whether a derived training/evaluation view is
    required before execution
- **Validation rules**:
  - `model_family` must be unique within the approved Phase 1 baseline set
  - trainable detector families must define both `default_variant` and `fallback_variant`
- **Relationships**:
  - referenced by `DetectorDatasetView`
  - referenced by `BaselineRun`

## Entity: BenchmarkManifest

- **Purpose**: Authoritative record of accepted assets, canonical classes, provenance, and
  split assignments for all Phase 1 baselines.
- **Fields**:
  - `manifest_id`: Stable benchmark identifier
  - `ontology_version`: Canonical household-object ontology version
  - `source_ids`: Included curated real-data sources
  - `split_versions`: Locked split identifiers for `train`, `val`, and `test_real_heldout`
  - `asset_counts`: Per-split and per-class asset totals
  - `classes`: Active canonical classes
  - `assets`: Accepted assets with normalized annotations
  - `created_at`: Manifest creation time
- **Validation rules**:
  - the manifest remains the sole source of truth for class membership and split assignment
  - every derived detector view must preserve the manifest's split and class totals
- **Relationships**:
  - parent of `YOLOClassMapping`
  - parent of `DetectorDatasetView`
  - referenced by `BaselineRun`

## Entity: YOLOClassMapping

- **Purpose**: Stable mapping between canonical Phase 1 class identifiers and YOLO class
  indices.
- **Fields**:
  - `mapping_id`: Stable mapping identifier
  - `manifest_id`: Reference to the authoritative benchmark manifest
  - `class_order`: Ordered list of canonical class IDs used to assign YOLO class indices
  - `index_by_class_id`: Zero-based lookup from canonical class ID to YOLO class index
  - `created_at`: Mapping creation time
- **Validation rules**:
  - `class_order` must contain every active canonical class exactly once
  - `index_by_class_id` values must be contiguous zero-based integers
  - all exported train, validation, and held-out test labels must use the same mapping
- **Relationships**:
  - child of `BenchmarkManifest`
  - referenced by `DetectorDatasetView`

## Entity: DetectorDatasetView

- **Purpose**: Reproducible detector-training and evaluation export derived from the
  benchmark manifest for YOLO11 execution.
- **Fields**:
  - `view_id`: Stable dataset-view identifier
  - `manifest_id`: Reference to the authoritative benchmark manifest
  - `model_family`: Baseline family using the view, initially `yolo11`
  - `model_variant`: Default model weights variant for the export
  - `dataset_root`: Root directory containing exported images, labels, and YAML metadata
  - `dataset_yaml_path`: Path to the Ultralytics dataset YAML file
  - `annotation_format`: Exported label format, `yolo_txt_normalized_xywh`
  - `class_mapping_id`: Reference to the `YOLOClassMapping`
  - `split_exports`: Export metadata for `train`, `val`, and `test_real_heldout`
  - `created_at`: View creation time
- **Validation rules**:
  - `manifest_id` must resolve to an existing benchmark manifest
  - exported split counts must match the manifest's asset counts
  - bounding boxes must be normalized into `[0, 1]` `xywh` coordinates
  - `test_real_heldout` may be exported for evaluation, but must never be designated as a
    training split
- **Relationships**:
  - child of `BenchmarkManifest`
  - references `YOLOClassMapping`
  - referenced by `BaselineRun`

## Entity: BaselineRun

- **Purpose**: One completed, blocked, or failed execution attempt for a baseline family on
  the shared benchmark.
- **Fields**:
  - `run_id`: Stable run identifier
  - `model_family`: Approved baseline family identifier
  - `model_variant`: Concrete executable variant such as `yolo11s`
  - `run_mode`: `zero_shot`, `trainable`, or `optional_stretch`
  - `manifest_id`: Reference to the benchmark manifest
  - `dataset_view_id`: Optional reference to a derived detector dataset view
  - `resolution`: Evaluation or training image size
  - `precision_mode`: Numeric precision mode
  - `batch_size`: Effective batch size
  - `seed`: Optional random seed
  - `status`: `planned`, `running`, `completed`, `blocked`, or `failed`
  - `hardware_profile`: Captured hardware summary
  - `metrics`: Quantitative output values
  - `artifact_root`: Root directory for exported failures or training artifacts
  - `started_at`: Start time
  - `ended_at`: End time
- **Validation rules**:
  - YOLO11 runs must reference a valid `dataset_view_id`
  - completed runs must include aggregate quality metrics, latency, and peak VRAM
  - blocked or failed runs must still record execution configuration and explanatory notes
  - `test_real_heldout` must never be used for training, tuning, or checkpoint selection
- **State transitions**:
  - `planned -> running`
  - `running -> completed`
  - `running -> blocked`
  - `running -> failed`
- **Relationships**:
  - references `BaselineFamily`
  - references `BenchmarkManifest`
  - optionally references `DetectorDatasetView`
  - parent of `FailureExample`

## Entity: FailureExample

- **Purpose**: Representative qualitative error used for detector and baseline comparison
  review.
- **Fields**:
  - `failure_id`: Stable failure identifier
  - `run_id`: Reference to the baseline run
  - `asset_id`: Reference to the originating benchmark asset
  - `failure_type`: Canonical failure bucket such as `occlusion`, `lighting`,
    `background_confusion`, `scale_distance`, `class_ambiguity`,
    `annotation_inconsistency`, or `runtime_constraint`
  - `expected_class_id`: Optional expected canonical class ID
  - `predicted_class_id`: Optional predicted canonical class ID
  - `artifact_path`: Optional visualization path
  - `notes`: Analyst note
- **Validation rules**:
  - `failure_type` must use the shared taxonomy
  - referenced `asset_id` must belong to the run's benchmark manifest
- **Relationships**:
  - child of `BaselineRun`
  - references `BenchmarkManifest`

## Entity: EvaluationReport

- **Purpose**: Comparison-ready summary across the approved Phase 1 baseline families.
- **Fields**:
  - `report_id`: Stable summary identifier
  - `manifest_id`: Shared benchmark manifest identifier
  - `run_ids`: Included baseline runs
  - `summary_metrics`: Aggregate metrics by baseline family
  - `top_failures`: Ranked failure categories
  - `recommended_path`: Preferred baseline family after comparison
  - `blocked_items`: Deferred or blocked runs
  - `created_at`: Summary creation time
- **Validation rules**:
  - all included runs must reference the same benchmark manifest
  - the recommended path must reference at least one completed required baseline
  - blocked runs must remain visible in the final summary rather than disappearing from the
    comparison record
- **Relationships**:
  - aggregates `BaselineRun`
  - aggregates `FailureExample`
