# Quickstart: Phase 1 Baseline Implementation

This quickstart describes the intended Phase 1 workflow after implementation.

## 1. Create the project environment

```powershell
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Expected result:

- a reproducible local Python environment exists for the project
- the runtime is compatible with the Phase 1 ML dependencies

## 2. Configure the ontology and source selection

Prepare a configuration file that defines:

- the approved household-object classes
- source priority order
- source-label to canonical-label mappings
- split policy for `train`, `val`, and `test_real_heldout`

Expected result:

- one fixed ontology is ready for benchmark preparation
- source mappings are explicit before data ingestion begins

## 3. Prepare the benchmark manifest

```powershell
python -m src.cli.prepare_benchmark --config config\phase1.yaml --output artifacts\manifests\phase1-benchmark.json
```

Expected result:

- curated real-image assets are normalized into the shared schema
- the held-out real test split is frozen
- a benchmark manifest is written with per-split and per-class counts

## 4. Validate benchmark integrity

```powershell
pytest tests\schema tests\integration -k benchmark
```

Expected result:

- split leakage checks pass
- schema validation passes for the benchmark manifest
- duplicate or unmapped assets are reported before baseline execution

## 5. Run the zero-shot reference baseline

```powershell
python -m src.cli.run_baseline --model grounding_dino --manifest artifacts\manifests\phase1-benchmark.json --report artifacts\reports\grounding-dino.json
```

Expected result:

- zero-shot evaluation finishes on the local workstation
- runtime, VRAM, aggregate metrics, and representative failures are captured

## 6. Run the compact trainable baseline

```powershell
python -m src.cli.run_baseline --model florence2 --manifest artifacts\manifests\phase1-benchmark.json --report artifacts\reports\florence2.json
```

Expected result:

- one required trainable baseline completes locally
- the report includes comparable metrics to the zero-shot reference baseline

## 7. Generate the Phase 1 summary

```powershell
python -m src.cli.summarize_phase1 --reports artifacts\reports --output artifacts\reports\phase1-summary.json
```

Expected result:

- the project has a comparison-ready evaluation summary
- the summary names the preferred baseline path and top failure categories

## Optional gated step

Only after the required baseline gates pass, an optional stretch-model run may be attempted.
This step is not required for the Phase 1 MVP.
