# Robotic Vision

This repository is being developed for **CIS 630: Advanced Artificial Intelligence Systems**.

Phase 1 current results
The current repository state includes a validated Phase 1 scaffold and live real-data baseline runs.

benchmark preparation completed successfully and produced artifacts/manifests/phase1-benchmark.json
the current benchmark uses manifest_id: phase1-local, one source (coco2017), and two active classes (mug, book)
both baseline commands run successfully at the CLI level and emit valid structured reports
the florence-2 training and evaluation path is wired and runs locally
latest held-out comparison shows that the VLM path (florence-2) performed worse than yolo11 on detection metrics
local validation passed with 7/7 pytest checks green

## Overview

This repository defines a hardware-aware roadmap for a robotic vision project focused on **household object detection in realistic indoor scenes**. The intended workflow is:

1. Build a strong real-image baseline.
2. Identify failure modes on real data.
3. Generate targeted synthetic data for those failures.
4. Retrain with mixed real and synthetic data.
5. Validate only on held-out real-world data.

The goal is not to build the biggest possible model. The goal is to build a **reproducible, measurable, local-first pipeline** that can later scale to larger hardware if needed.

## Confirmed Hardware Constraints

Hardware detected on this workstation:

- **GPU:** NVIDIA GeForce RTX 5070, 12 GB VRAM (`12227 MiB`)
- **CPU:** AMD Ryzen 9 9900X, 12 cores / 24 threads
- **RAM:** 32 GB

These constraints are good for:

- compact or mid-sized vision models
- mixed-precision training
- LoRA / QLoRA-style adaptation
- moderate-resolution object detection experiments
- small synthetic scene generation runs

These constraints are **not** ideal for:

- full fine-tuning of multi-billion-parameter VLMs
- large multi-camera simulation workloads
- complex Isaac Sim scenes rendered at high resolution
- full local ingestion of very large public datasets on day one

## Getting Started

### Prerequisites

- Python 3.11 (managed via [uv](https://github.com/astral-sh/uv))
- NVIDIA GPU with ≥ 8 GB VRAM (RTX 5070 12 GB is the reference platform)
- CUDA-compatible drivers installed

### 1. Create the Environment

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1

# Core dependencies only (CLI, schemas, validation)
uv pip install -e ".[dev]"

# Add ML dependencies when you are ready to run models
uv pip install -e ".[dev,ml]"
```

The `[ml]` group installs PyTorch, Transformers, Ultralytics, pycocotools, and the rest of
the model stack. Omit it during initial setup if you only want to test the pipeline with
fixture data.

### 2. Download the Datasets

The pipeline consumes COCO-format JSON annotation files. You supply these yourself and point
the config at them.

**COCO 2017** — recommended starting point

Download from [https://cocodataset.org/#download](https://cocodataset.org/#download):


| File                           | Size    | Use                            |
| ------------------------------ | ------- | ------------------------------ |
| `val2017.zip`                  | ~1 GB   | Quick dry run                  |
| `train2017.zip`                | ~18 GB  | Full training set              |
| `annotations_trainval2017.zip` | ~241 MB | Required annotation JSON files |


Extract so that your layout looks like:

```
data/coco2017/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── images/
    ├── train2017/
    └── val2017/
```

**Open Images V7** — optional household-object supplement

Use the [FiftyOne dataset zoo](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html)
or the [OIDv4 toolkit](https://github.com/EscVM/OIDv4_ToolKit) to pull targeted subsets.
Only download the categories that match your household ontology — ingesting the full Open
Images dataset locally is not recommended on this hardware.

### 3. Configure the Ontology and Sources

Edit `config/phase1.yaml` before running any pipeline step.

The two sections that require your input:

```yaml
ontology:
  - class_id: mug
    canonical_name: mug
    aliases: [cup]          # labels that map to this class in your datasets
  - class_id: bottle
    canonical_name: bottle
    aliases: [water bottle]
  # add the remaining classes from the recommended list below

sources:
  - source_id: coco2017
    name: COCO 2017
    source_type: public_curated
    format: coco
    path: data/coco2017/annotations/instances_train2017.json

  - source_id: open_images_v7   # optional
    name: Open Images V7 subset
    source_type: public_curated
    format: open_images
    path: data/open_images/annotations.json
```

All other settings (split ratios, model variants, image size, precision mode) have
hardware-appropriate defaults for the RTX 5070 and do not need to be changed for a first run.

**Recommended starting ontology** (10–15 classes):
`mug, bottle, bowl, plate, spoon, fork, phone, remote, book, backpack, chair, table, lamp, box, trash can`

Start with a small, tight class set. A narrower ontology makes label cleanup and early
fine-tuning significantly more tractable on local hardware.

### 4. Validate the Setup

Run the test suite before touching any real data. All checks should pass against the bundled
fixture data without requiring model downloads.

```powershell
pytest -q
```

Expected output: `7 passed` (schema validation, split-leakage checks, report format checks).

### 5. Prepare the Benchmark Manifest

```powershell
phase1 prepare-benchmark `
  --config config\phase1.yaml `
  --output artifacts\manifests\phase1-benchmark.json
```

This step normalises your data sources into a single authoritative manifest: one class
ontology, one box convention, and one frozen split policy. All downstream steps consume this
manifest rather than the raw source files.

Run the schema checks after preparation:

```powershell
pytest tests\schema tests\integration -k benchmark
```

### 6. Export the YOLO Detector View

```powershell
phase1 export-yolo-view `
  --manifest artifacts\manifests\phase1-benchmark.json `
  --output-dir artifacts\datasets\yolo11 `
  --metadata artifacts\datasets\yolo11\view.json
```

This derives a YOLO-format dataset (dataset YAML + normalised `xywh` label files) from the
benchmark manifest. It must be run before the YOLO11 baseline step. The Grounding DINO and
Florence-2 baselines consume the manifest directly and do not require this step.

### 7. Run the Baselines

Run each baseline in order. Each produces a structured JSON report under `artifacts/reports/`.

```powershell
# Zero-shot reference
phase1 run-baseline `
  --model grounding_dino `
  --manifest artifacts\manifests\phase1-benchmark.json `
  --report artifacts\reports\grounding-dino.json

# Compact trainable baseline
phase1 run-baseline `
  --model florence2 `
  --manifest artifacts\manifests\phase1-benchmark.json `
  --report artifacts\reports\florence2.json

# Detector validation baseline
phase1 run-baseline `
  --model yolo11 `
  --manifest artifacts\manifests\phase1-benchmark.json `
  --dataset-view artifacts\datasets\yolo11\view.json `
  --report artifacts\reports\yolo11.json
```

> **Note:** The Grounding DINO and Florence-2 runners currently return `status: blocked`
> until the model inference wiring is added. The YOLO11 runner downloads weights
> automatically via Ultralytics and is the most complete path for an initial end-to-end run.
> Blocked runs still produce valid, schema-compliant reports that can flow into the summary
> step.

### 8. Generate the Phase 1 Summary

```powershell
phase1 summarize-phase1 `
  --reports artifacts\reports `
  --output artifacts\reports\phase1-summary.json
```

The summary compares all available baseline reports, names the preferred path by mAP, and
lists the top failure categories for Phase 2 planning.

If the `phase1` entry point is unavailable, replace it with the module form:

```powershell
python -m src.cli.main <command> [options]
```

---

## Methodology Validity Summary

The overall methodology is valid, but it needs to be scoped carefully for this hardware.

### What is valid now

- **Real-data baseline first:** this is the right starting point.
- **Grounding DINO as a zero-shot reference:** useful as a no-training baseline.
- **Florence-2 Base as a trainable local baseline:** realistic for this GPU.
- **Synthetic data as a targeted supplement, not a replacement:** correct sim-to-real strategy.
- **Frozen held-out real test set:** essential and already correctly planned.

### What should be treated as a stretch goal

- **PaliGemma 2 3B + LoRA:** feasible only as a constrained experiment, not the default core track.
Use 4-bit loading or another memory-saving setup, low image resolution, batch size 1, and gradient accumulation.
- **Isaac Sim / Omniverse synthetic generation:** feasible only in a lightweight form on this machine.
The latest Isaac Sim releases officially target higher-end GPUs than a 12 GB card, so the sim pipeline should start small and may need an older version, headless generation, or a fallback synthetic pipeline if performance is poor.

### What should be changed from the original roadmap

- Do **not** start by normalizing all of Open Images or Objects365 locally.
Start with curated subsets plus smaller indoor datasets.
- Do **not** run all model tracks in parallel at full scale.
Sequence them: zero-shot baseline, one trainable compact model, then one optional stretch model.
- Do **not** assume the newest Isaac Sim stack will be comfortable on this machine.
Treat synthetic generation as a scoped milestone with a fallback plan.

## Recommended Project Scope

This project should remain a **2D household object detection** project for the first full cycle.

Recommended target classes for the first iteration:

- mug
- bottle
- bowl
- plate
- spoon
- fork
- phone
- remote
- book
- backpack
- chair
- table
- lamp
- box
- trash can

Keep the ontology intentionally small at first. A tighter class set makes dataset cleanup, fine-tuning, and evaluation much more realistic on local hardware.

## Recommended Model Strategy

### Track A: Zero-shot reference

- **Grounding DINO Base or Tiny**
- Purpose: establish a no-training baseline and identify class-level failure cases quickly
- Cost: low compared with training

### Track B: Primary trainable baseline

- **Florence-2 Base**
- Purpose: main compact model for local experimentation
- Why it fits: small enough to iterate locally, supports detection-style prompting, and is practical for lightweight fine-tuning compared with larger VLMs

### Track C: Detector validation baseline

- **YOLO11 Small (`yolo11s`) with `yolo11n` fallback**
- Purpose: add a detector-oriented Phase 1 comparison track without changing the shared ingestion contract
- Why it fits: compact YOLO11 variants are realistic for a 12 GB workstation and keep detector training/export decisions explicit before later synthetic-data work

### Track D: Optional stretch experiment

- **PaliGemma 2 3B with LoRA / QLoRA**
- Purpose: test whether a compact VLM improves robustness enough to justify the extra complexity
- Constraint: this should be optional, not required for project success

The detector path is now part of the Phase 1 baseline package. Future synthetic-data work should extend the same detector-view export contract instead of creating a parallel ingestion flow.

## Dataset Strategy

### Recommended data sources

- **Curated subsets of Open Images or Objects365**
Use only categories and image slices that match the household ontology.
- **Indoor or household-focused datasets from Roboflow Universe, Hugging Face, or academic sources**
Use these for class relevance and cluttered indoor scenes.
- **In-house collected images**
These are the most important for final validation.

### Label normalization rules

All imported data should be converted into one unified schema:

- one annotation format
- one class list
- one box convention
- one split policy

### Split policy

Create and freeze:

- `train`
- `val`
- `test_real_heldout`

The held-out real test set must never be used for model selection after Phase 1.

## Four-Phase Methodology

### Phase 1: Real-data baseline

Objective:
Build a real-world baseline before any simulation work.

Tasks:

- finalize the class ontology
- collect a manageable real-image dataset
- normalize labels
- run Grounding DINO as a zero-shot baseline
- fine-tune Florence-2 Base locally
- export a reproducible YOLO detector view from the benchmark manifest
- run YOLO11 with `yolo11s` as the default and `yolo11n` as the hardware-fit fallback
- log failure cases by cause

Track at minimum:

- mAP
- precision / recall
- per-class AP
- latency per image
- GPU memory usage
- qualitative failure examples

Hardware-aware guidance:

- start with moderate image sizes such as `640` or `768`
- use mixed precision
- prefer frozen-backbone or parameter-efficient tuning first
- keep batch size small and scale with gradient accumulation

### Phase 1 current results

The current repository state includes a validated Phase 1 scaffold and dry run for the
real-data baseline workflow.

- benchmark preparation completed successfully and produced `artifacts/manifests/phase1-benchmark.json`
- the current sample benchmark uses `manifest_id: phase1-local`, one source (`coco2017`),
two accepted assets, and two active classes (`mug`, `book`)
- both baseline commands run successfully at the CLI level and emit valid structured reports
- the Florence-2 path now runs live with full training plus held-out evaluation
- latest held-out model comparison shows Florence-2 (VLM) underperforming YOLO11 on precision
  and mAP for the current local setup
- local validation passed with `7/7` pytest checks green

This means Phase 1 is implemented as a reproducible benchmark-and-reporting pipeline, even
though full live model execution is still gated on wiring the actual predictors.

### Phase 2: Lightweight synthetic data generation

Objective:
Generate synthetic scenes that target the failure modes discovered in Phase 1.

Synthetic scope for this machine:

- one or two scene archetypes only
- one RGB camera
- 2D bounding boxes as the primary output
- limited clutter and limited sensor count
- pilot exports before large runs

Suggested first scenes:

- kitchen counter / tabletop
- shelf or desk clutter scene

Randomize:

- object pose
- clutter density
- distractor objects
- lighting strength and direction
- small camera viewpoint shifts
- textures and materials where reasonable

Important constraint:
The newest Isaac Sim releases officially recommend at least a 16 GB GPU. With a 12 GB card, complex scenes, many sensors, and high-resolution rendering are risky. This means the synthetic phase is valid, but only if it begins as a lightweight pipeline with a fallback plan.

Fallback if Isaac Sim is unstable:

- pin to an older, lighter Isaac Sim setup
- use headless export rather than interactive heavy scenes
- reduce scene complexity and render resolution
- switch to a simpler synthetic data pipeline for 2D detection

### Phase 3: Mixed retraining

Objective:
Test whether synthetic data improves real-world performance.

Required experiments:

1. real-only baseline rerun
2. synthetic-only training
3. mixed real + synthetic training
4. targeted synthetic oversampling for failure cases

Selection rule:
Choose the final checkpoint using **real validation data only**.

Recommended order under this hardware:

- first retrain Florence-2 Base with mixed data
- only run the PaliGemma experiment if Phase 1 and Phase 2 are already stable

### Phase 4: Real-world validation

Objective:
Measure whether the final detector is usable outside the training distribution.

Validation data should include:

- daylight scenes
- warm indoor lighting
- low-light scenes
- cluttered scenes
- partial occlusions
- small or distant objects
- reflective or transparent objects where possible
- images from the intended deployment camera if available

Track:

- mAP on held-out real data
- precision / recall
- per-class miss rate
- false positives on unseen backgrounds
- latency / FPS on deployment hardware
- stability on short video clips

## Failure Taxonomy

Every major error should be assigned to one of the following buckets:

- occlusion failure
- lighting mismatch
- background confusion
- scale or distance issue
- class ambiguity
- annotation inconsistency
- simulator artifact transfer
- runtime or latency constraint

This makes later improvement decisions much easier.

## Practical Training Budget

For this workstation, the project should assume:

- **baseline training:** fully feasible
- **compact-model fine-tuning:** feasible
- **3B VLM adaptation:** possible, but only with memory-saving techniques
- **synthetic generation:** feasible only if scene scope stays small

Recommended default policy:

- prioritize one strong compact baseline over many partially finished tracks
- log VRAM, wall-clock time, and throughput in every run
- keep experiments reproducible and small before scaling up

## Deliverables

Expected deliverables across the project:

- unified dataset schema for household objects
- baseline model comparison report
- training and evaluation scripts
- fixed held-out real validation and test splits
- lightweight synthetic data generation pipeline
- mixed-data ablation study
- final real-world validation report

## Success Criteria

This project is successful if it demonstrates:

- a reproducible real-data baseline
- a valid, lightweight synthetic data pipeline
- a measurable real-world improvement from synthetic data or a clear result showing when it does not help
- a documented understanding of the remaining failure modes under local hardware limits

## Recommended Timeline

- **Weeks 1-2:** ontology, dataset curation, zero-shot baseline
- **Weeks 3-4:** Florence-2 fine-tuning and error analysis
- **Weeks 5-6:** pilot synthetic scene generation
- **Weeks 7-8:** mixed retraining and ablations
- **Weeks 9-10:** real-world validation and final report

## Reference Checks

The hardware-scoping decisions in this README were cross-checked against:

- [NVIDIA Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- [Grounding DINO model card](https://huggingface.co/IDEA-Research/grounding-dino-base)
- [Florence-2 Base model card](https://huggingface.co/microsoft/Florence-2-base)
- [Hugging Face Florence-2 fine-tuning guide](https://huggingface.co/blog/finetune-florence2)
- [Hugging Face PaliGemma documentation](https://huggingface.co/docs/transformers/model_doc/paligemma)

## Bottom Line

This project plan is **methodologically sound**, but only if it is executed as a **compact, staged detection project** rather than a large-scale VLM or simulation effort.

The most realistic path on this hardware is:

1. Grounding DINO zero-shot baseline
2. Florence-2 Base as the main trainable model
3. small, targeted synthetic generation
4. mixed retraining evaluated only on real held-out data
5. PaliGemma 2 3B only as an optional stretch experiment

That version of the roadmap is ambitious enough to be meaningful, while still realistic on an RTX 5070 with 12 GB VRAM and 32 GB RAM.

## Local Implementation Status

The repository now includes a Phase 1 implementation scaffold under `src/` and `tests/`
covering:

- benchmark preparation from curated real-image sources
- benchmark manifest validation and split-leakage checks
- baseline run report validation and summary generation
- fixture-backed integration tests for benchmark preparation, baseline execution, and Phase 1 summary generation

The current model runners are intentionally safe by default: they return a documented
`blocked` status unless explicit model wiring is provided. This keeps local dry runs and
tests reproducible without forcing heavyweight model downloads during repository bootstrap.

The installed CLI entry point is `phase1`, with equivalent module access through
`python -m src.cli.main`.

## Phase 1 File Guide

The main Phase 1 artifacts live under `specs/001-phase1-baseline/` and are organized so the
project can be understood from requirements through implementation.

- `specs/001-phase1-baseline/spec.md`
defines the Phase 1 feature scope, user stories, and success criteria
- `specs/001-phase1-baseline/plan.md`
captures the implementation plan, technical context, hardware constraints, and structure
- `specs/001-phase1-baseline/research.md`
records the key decisions behind model choice, dataset order, and local-environment strategy
- `specs/001-phase1-baseline/data-model.md`
describes the core entities such as benchmark manifests, assets, runs, metrics, and failures
- `specs/001-phase1-baseline/contracts/`
contains the CLI contract and the JSON schemas for benchmark manifests and run reports
- `specs/001-phase1-baseline/tasks.md`
is the executable checklist used to implement and verify the Phase 1 work
- `config/phase1.yaml`
is the sample local configuration for ontology and source selection
- `src/cli/prepare_benchmark.py`
builds the normalized benchmark manifest from approved real-image inputs
- `src/cli/run_baseline.py`
executes a baseline path and writes the structured run report
- `src/cli/summarize_phase1.py`
compares completed runs and generates the Phase 1 summary artifact
- `artifacts/reports/phase1-dry-run.md`
records the implementation dry run, what succeeded, and what is still intentionally blocked

Together, these files provide the complete Phase 1 trail: what the project is supposed to do,
how it is designed, how it was implemented, and what the current results look like.