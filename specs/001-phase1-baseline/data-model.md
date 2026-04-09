# Data Model: Phase 1 Baseline Implementation

## Entity: HouseholdObjectClass

- **Purpose**: Canonical definition of an approved Phase 1 household category.
- **Fields**:
  - `class_id`: Stable identifier used across manifests and reports
  - `canonical_name`: Approved class name used in the benchmark
  - `aliases`: Accepted alternative names for source mapping
  - `source_mappings`: Map of source labels to the canonical class
  - `status`: `active` or `deferred`
- **Validation rules**:
  - `canonical_name` must be unique within the ontology
  - all `source_mappings` must resolve to exactly one canonical class
- **Relationships**:
  - referenced by `AnnotationRecord`
  - referenced by `MetricRecord`

## Entity: DatasetSource

- **Purpose**: Metadata for an approved labeled-data source.
- **Fields**:
  - `source_id`: Stable identifier
  - `name`: Human-readable source name
  - `source_type`: `public_curated`, `community_vetted`, or `in_house`
  - `license_reference`: Source license or usage note
  - `priority_order`: Extraction order within Phase 1
  - `notes`: Curation or quality caveats
- **Validation rules**:
  - every accepted asset must reference one approved `source_id`
- **Relationships**:
  - parent of `DatasetAsset`

## Entity: DatasetAsset

- **Purpose**: One accepted real-image example included in the benchmark.
- **Fields**:
  - `asset_id`: Stable identifier
  - `source_id`: Reference to `DatasetSource`
  - `original_identifier`: Source-native image identifier
  - `relative_path`: Repository or artifact-relative path to the image
  - `width`: Image width in pixels
  - `height`: Image height in pixels
  - `split_name`: `train`, `val`, or `test_real_heldout`
  - `content_hash`: Hash used for duplicate detection
  - `review_status`: `accepted`, `rejected`, or `held_for_review`
- **Validation rules**:
  - each `asset_id` must appear in exactly one split
  - `content_hash` duplicates across splits must be flagged
  - only `accepted` assets may appear in the benchmark manifest
- **Relationships**:
  - parent of `AnnotationRecord`
  - included in `BenchmarkManifest`
  - referenced by `FailureExample`

## Entity: AnnotationRecord

- **Purpose**: Normalized object annotation for a benchmark asset.
- **Fields**:
  - `annotation_id`: Stable identifier
  - `asset_id`: Reference to `DatasetAsset`
  - `class_id`: Reference to `HouseholdObjectClass`
  - `source_label`: Original source label
  - `bbox_xyxy`: Bounding box in normalized benchmark format
  - `is_ignored`: Boolean for excluded annotations
- **Validation rules**:
  - `bbox_xyxy` must lie within image bounds
  - `class_id` must resolve to an active ontology class
  - ignored annotations must not be counted in evaluation metrics
- **Relationships**:
  - child of `DatasetAsset`
  - references `HouseholdObjectClass`

## Entity: SplitDefinition

- **Purpose**: Rules and metadata for one benchmark split.
- **Fields**:
  - `split_name`: `train`, `val`, or `test_real_heldout`
  - `purpose`: Description of intended use
  - `selection_rules`: Human-readable split policy
  - `locked_at`: Timestamp when the split was frozen
  - `version`: Split-policy version string
- **Validation rules**:
  - `test_real_heldout` must be locked before model selection
  - one and only one record must exist for each required split
- **Relationships**:
  - governs `DatasetAsset.split_name`
  - referenced by `BenchmarkManifest`

## Entity: BenchmarkManifest

- **Purpose**: Versioned description of the benchmark used for a set of runs.
- **Fields**:
  - `manifest_id`: Stable identifier
  - `ontology_version`: Ontology version string
  - `source_ids`: Included dataset sources
  - `split_versions`: References to the active split definitions
  - `asset_counts`: Per-split and per-class counts
  - `created_at`: Manifest creation time
- **Validation rules**:
  - manifest counts must match included accepted assets
  - manifest must reference the frozen held-out real test split
- **Relationships**:
  - aggregates `DatasetAsset`, `HouseholdObjectClass`, and `SplitDefinition`
  - referenced by `BaselineRun`

## Entity: BaselineRun

- **Purpose**: One benchmark execution for a baseline model path.
- **Fields**:
  - `run_id`: Stable identifier
  - `model_family`: `grounding_dino`, `florence2`, or approved optional value
  - `run_mode`: `zero_shot`, `trainable`, or `optional_stretch`
  - `manifest_id`: Reference to `BenchmarkManifest`
  - `resolution`: Evaluation or training image size
  - `precision_mode`: Numeric precision setting
  - `batch_size`: Effective batch size
  - `seed`: Optional random seed
  - `hardware_profile`: Captured hardware summary
  - `status`: `planned`, `running`, `completed`, `blocked`, or `failed`
  - `started_at`: Start time
  - `ended_at`: End time
- **Validation rules**:
  - `status` must follow valid transitions
  - `manifest_id` must reference a valid benchmark manifest
  - required metrics are mandatory only for `completed` runs
- **State transitions**:
  - `planned -> running`
  - `running -> completed`
  - `running -> blocked`
  - `running -> failed`
- **Relationships**:
  - parent of `MetricRecord`
  - parent of `FailureExample`
  - summarized by `EvaluationReport`

## Entity: MetricRecord

- **Purpose**: Quantitative output from a completed baseline run.
- **Fields**:
  - `run_id`: Reference to `BaselineRun`
  - `split_name`: Evaluated split
  - `metric_name`: e.g. `mAP`, `precision`, `recall`, `latency_ms`, `peak_vram_mb`
  - `metric_value`: Numeric value
  - `class_id`: Optional class reference for per-class metrics
- **Validation rules**:
  - required aggregate metrics must exist for every completed run
  - `class_id` is mandatory only for class-level metrics
- **Relationships**:
  - child of `BaselineRun`
  - optionally references `HouseholdObjectClass`

## Entity: FailureExample

- **Purpose**: Representative qualitative error for analysis and review.
- **Fields**:
  - `failure_id`: Stable identifier
  - `run_id`: Reference to `BaselineRun`
  - `asset_id`: Reference to `DatasetAsset`
  - `failure_type`: `occlusion`, `lighting`, `background_confusion`, `scale_distance`,
    `class_ambiguity`, `annotation_inconsistency`, or `runtime_constraint`
  - `expected_class_id`: Optional expected class
  - `predicted_class_id`: Optional predicted class
  - `artifact_path`: Path to saved visualization or crop
  - `notes`: Analyst note
- **Validation rules**:
  - `failure_type` must use the agreed taxonomy
  - `artifact_path` must exist for accepted failure examples
- **Relationships**:
  - child of `BaselineRun`
  - references `DatasetAsset`

## Entity: EvaluationReport

- **Purpose**: Comparison-ready summary for one or more baseline runs.
- **Fields**:
  - `report_id`: Stable identifier
  - `manifest_id`: Reference to `BenchmarkManifest`
  - `run_ids`: Included runs
  - `summary_metrics`: Aggregate comparison values
  - `top_failures`: Ranked failure categories
  - `recommended_path`: Preferred baseline direction
  - `blocked_items`: Deferred or infeasible experiments
  - `created_at`: Report creation time
- **Validation rules**:
  - every `run_id` must reference a run using the same benchmark manifest
  - the recommended path must cite at least one completed required run
- **Relationships**:
  - aggregates `BaselineRun` and `FailureExample`
