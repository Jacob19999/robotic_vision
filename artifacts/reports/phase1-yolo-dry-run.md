# Phase 1 YOLO Dry Run

Date: 2026-04-15

## Commands

```bash
uv run python -m src.cli.main export-yolo-view \
  --manifest tests/fixtures/phase1_yolo/sample_benchmark_manifest.json \
  --output-dir artifacts/datasets/yolo11-dry-run \
  --metadata artifacts/datasets/yolo11-dry-run/view.json

uv run python -m src.cli.main run-baseline \
  --model yolo11 \
  --manifest tests/fixtures/phase1_yolo/sample_benchmark_manifest.json \
  --dataset-view artifacts/datasets/yolo11-dry-run/view.json \
  --report artifacts/reports/yolo11-dry-run.json
```

## Outcome

- `export-yolo-view` completed and wrote a detector-view metadata file plus YOLO-format split exports under `artifacts/datasets/yolo11-dry-run/`
- `run-baseline --model yolo11` completed the CLI workflow and wrote a structured blocked report to `artifacts/reports/yolo11-dry-run.json`
- Report status: `blocked`
- Blocking reason: `Ultralytics` is not installed in the current runtime, so the runner recorded the missing optional dependency instead of emitting a misleading completed run

## Notes

- The dry run confirms the new detector-view export path, manifest/view validation, and run-report generation end to end
- The repository-level entrypoint `uv run phase1 ...` did not resolve `src` in this environment, so the documented module form `uv run python -m src.cli.main ...` was used instead
- Once the `ml` extra is installed with `Ultralytics`, the same workflow can be rerun without changing the manifest or detector-view contract
