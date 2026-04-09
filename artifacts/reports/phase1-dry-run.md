# Phase 1 Dry Run

Date: 2026-04-08

## Scope

This dry run verified the implemented Phase 1 CLI surface, schema validation, fixture-backed
benchmark preparation, blocked-safe baseline execution, and summary generation.

## Commands Run

```powershell
python -m src.cli.main prepare-benchmark --config config/phase1.yaml --output artifacts/manifests/phase1-benchmark.json
python -m src.cli.main run-baseline --model grounding_dino --manifest artifacts/manifests/phase1-benchmark.json --report artifacts/reports/grounding-dino.json
python -m src.cli.main run-baseline --model florence2 --manifest artifacts/manifests/phase1-benchmark.json --report artifacts/reports/florence2.json
python -m src.cli.main summarize-phase1 --reports artifacts/reports --output artifacts/reports/phase1-summary.json
python -m src.cli.main summarize-phase1 --reports artifacts/reports/sample-summary-inputs --output artifacts/reports/phase1-summary.json
python -m pytest -q
```

## Results

- Benchmark preparation succeeded and wrote `artifacts/manifests/phase1-benchmark.json`.
- The generated manifest used `manifest_id: phase1-local`, one source (`coco2017`), and two accepted assets.
- Both baseline commands succeeded operationally and produced valid structured reports.
- The Grounding DINO report was `blocked` with hardware detected as `NVIDIA GeForce RTX 5070`, `12227 MB` VRAM, and `32360 MB` system RAM.
- The Florence-2 report was `blocked` with the same hardware profile.
- The first summary attempt against the blocked live reports failed as designed because no completed required baseline report was available.
- The summary command succeeded against fixture-backed completed reports and recommended `florence2` over `grounding_dino` (`mAP 0.62` vs `0.45`).
- `pytest` passed with `7/7` tests green.

## Notes

- The sample config path needed to be corrected to resolve fixture data relative to `config/phase1.yaml`.
- Benchmark manifest serialization now excludes internal `None` fields so the emitted contract matches the published schema.
- The current model runners are intentionally blocked by default until explicit predictor/model wiring is added. This keeps repository bootstrap and tests lightweight while preserving the completed-run path through fixtures and integration coverage.

## Follow-up

- Wire concrete Grounding DINO and Florence-2 predictors when the ML dependency group and model downloads are intentionally enabled.
- Re-run the same CLI flow on a real curated benchmark once local model execution is configured.
