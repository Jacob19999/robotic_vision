from __future__ import annotations

from pathlib import Path

from src.cli import run_baseline as run_baseline_module
from src.models.common.predictions import AssetPrediction, PredictedAnnotation, RunnerResult
from src.utils.paths import read_json


class _StubRunner:
    def run(self, manifest):  # noqa: ANN001
        predictions = []
        for asset in manifest.assets:
            predictions.append(
                AssetPrediction(
                    asset_id=asset.asset_id,
                    predictions=[
                        PredictedAnnotation(
                            class_id=asset.annotations[0].class_id,
                            bbox_xyxy=asset.annotations[0].bbox_xyxy,
                            score=0.99,
                        )
                    ],
                )
            )
        return RunnerResult(status="completed", predictions=predictions, notes="stub")


def test_run_baseline_writes_completed_report(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_reports" / "sample_run_input.json"
    report_path = tmp_path / "grounding-report.json"
    monkeypatch.setitem(run_baseline_module.RUNNER_FACTORIES, "grounding_dino", lambda: _StubRunner())

    report = run_baseline_module.run_baseline("grounding_dino", manifest_path, report_path)
    saved = read_json(report_path)

    assert report["status"] == "completed"
    assert saved["model_family"] == "grounding_dino"
    assert saved["metrics"]["mAP"] >= 0.5

