# Quickstart: Phase 1 YOLO Baseline Validation

This quickstart describes the intended Phase 1 workflow after implementation.

## 1. Create the project environment

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,ml]"
```

Expected result:

- a reproducible local Python environment exists for the project
- the runtime includes the detector and baseline dependencies needed for YOLO11 validation
- `config/phase1.yaml` supplies the baseline defaults, including `yolo11s` as the primary
  detector variant and `yolo11n` as the fallback

## 2. Configure the ontology and source selection

Prepare a configuration file that defines:

- the approved household-object classes
- source priority order
- source-label to canonical-label mappings
- split policy for `train`, `val`, and `test_real_heldout`
- source file paths relative to the config file location unless absolute paths are used

Expected result:

- one fixed ontology is ready for benchmark preparation
- source mappings are explicit before any detector-view export begins

## 3. Prepare the benchmark manifest

```powershell
phase1 prepare-benchmark --config config\phase1.yaml --output artifacts\manifests\phase1-benchmark.json
```

Expected result:

- curated real-image assets are normalized into the shared schema
- the held-out real test split is frozen
- one authoritative benchmark manifest is written for all baseline families

## 4. Validate benchmark integrity

```powershell
pytest tests\schema tests\integration -k benchmark
```

Expected result:

- split leakage checks pass
- benchmark schema validation passes
- duplicate or unmapped assets are reported before any model-specific export

## 5. Export the YOLO detector view

```powershell
phase1 export-yolo-view --manifest artifacts\manifests\phase1-benchmark.json --output-dir artifacts\datasets\yolo11 --metadata artifacts\datasets\yolo11\view.json
```

Expected result:

- the authoritative benchmark is converted into a YOLO-format dataset YAML plus label files
- canonical class IDs are mapped into one stable YOLO class order
- `train`, `val`, and `test_real_heldout` exports are documented in detector-view metadata

## 6. Run the zero-shot reference baseline

```powershell
phase1 run-baseline --model grounding_dino --manifest artifacts\manifests\phase1-benchmark.json --report artifacts\reports\grounding-dino.json
```

Expected result:

- zero-shot evaluation finishes on the local workstation
- runtime, VRAM, aggregate metrics, and representative failures are captured

## 7. Run the compact trainable baseline

```powershell
phase1 run-baseline --model florence2 --manifest artifacts\manifests\phase1-benchmark.json --report artifacts\reports\florence2.json
```

Expected result:

- the existing compact trainable baseline completes locally
- the report remains directly comparable to the zero-shot reference baseline

## 8. Run the YOLO11 detector baseline

```powershell
phase1 run-baseline --model yolo11 --manifest artifacts\manifests\phase1-benchmark.json --dataset-view artifacts\datasets\yolo11\view.json --report artifacts\reports\yolo11.json
```

Expected result:

- YOLO11 trains or validates using the exported detector view without redefining the
  benchmark splits
- the report captures execution configuration, runtime, VRAM, aggregate metrics, and
  representative failures
- blocked runs still produce a structured report with the limiting reason recorded
- the runner records when it falls back from `yolo11s` to `yolo11n` because of the local
  hardware budget

## 9. Generate the Phase 1 summary

```powershell
phase1 summarize-phase1 --reports artifacts\reports --output artifacts\reports\phase1-summary.json
```

Expected result:

- the project has a comparison-ready summary across Grounding DINO, Florence-2, and YOLO11
- the summary names the preferred baseline path, top failure categories, and any blocked
  items

If the editable-install entry point is unavailable, use the equivalent module form:

```powershell
python -m src.cli.main <command> [options]
```
