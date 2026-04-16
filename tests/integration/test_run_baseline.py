from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import run_baseline as run_baseline_module
from src.models.common.predictions import AssetPrediction, ExecutionConfig, PredictedAnnotation, RunnerResult
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
        return RunnerResult(
            status="completed",
            execution_config=ExecutionConfig(
                model_variant="grounding-dino-base",
                resolution=640,
                precision_mode="fp16",
                batch_size=1,
            ),
            predictions=predictions,
            notes="stub",
        )


class _StubYoloRunner:
    def __init__(self, dataset_view) -> None:  # noqa: ANN001
        self.dataset_view = dataset_view

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
                            score=0.95,
                        )
                    ],
                )
            )
        return RunnerResult(
            status="completed",
            execution_config=ExecutionConfig(
                model_variant="yolo11s",
                resolution=640,
                precision_mode="fp16",
                batch_size=1,
                seed=42,
                dataset_view_id=self.dataset_view.view_id,
                checkpoint_reference="yolo11s.pt",
            ),
            predictions=predictions,
            notes="stub yolo",
        )


class _BlockedYoloRunner:
    def __init__(self, dataset_view) -> None:  # noqa: ANN001
        self.dataset_view = dataset_view

    def run(self, manifest):  # noqa: ARG002, ANN001
        return RunnerResult(
            status="blocked",
            execution_config=ExecutionConfig(
                model_variant="yolo11n",
                resolution=640,
                precision_mode="fp16",
                batch_size=1,
                seed=42,
                dataset_view_id=self.dataset_view.view_id,
                checkpoint_reference="yolo11n.pt",
            ),
            notes="Blocked after fallback to yolo11n because the hardware budget was exceeded.",
        )


def test_run_baseline_writes_completed_report(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_reports" / "sample_run_input.json"
    report_path = tmp_path / "grounding-report.json"
    monkeypatch.setitem(run_baseline_module.RUNNER_FACTORIES, "grounding_dino", lambda **_: _StubRunner())

    report = run_baseline_module.run_baseline("grounding_dino", manifest_path, report_path)
    saved = read_json(report_path)

    assert report["status"] == "completed"
    assert saved["model_family"] == "grounding_dino"
    assert saved["metrics"]["mAP"] >= 0.5


def test_run_baseline_writes_yolo_report_with_dataset_view(
    repo_root: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json"
    dataset_view_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_detector_view.json"
    report_path = tmp_path / "yolo-report.json"
    monkeypatch.setitem(
        run_baseline_module.RUNNER_FACTORIES,
        "yolo11",
        lambda dataset_view=None, **_: _StubYoloRunner(dataset_view),
    )

    report = run_baseline_module.run_baseline("yolo11", manifest_path, report_path, dataset_view_path)
    saved = read_json(report_path)

    assert report["status"] == "completed"
    assert saved["model_family"] == "yolo11"
    assert saved["execution_config"]["dataset_view_id"] == "phase1-yolo-sample-yolo11-view"
    assert saved["metrics"]["mAP"] >= 0.5


def test_run_baseline_requires_dataset_view_for_yolo(repo_root: Path, tmp_path: Path) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json"
    report_path = tmp_path / "yolo-report.json"

    with pytest.raises(ValueError, match="--dataset-view is required"):
        run_baseline_module.run_baseline("yolo11", manifest_path, report_path)


def test_run_baseline_rejects_manifest_dataset_view_mismatch(repo_root: Path, tmp_path: Path) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json"
    source_view_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_detector_view.json"
    invalid_view_path = tmp_path / "bad-view.json"
    payload = read_json(source_view_path)
    payload["manifest_id"] = "wrong-manifest"
    invalid_view_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_id does not match"):
        run_baseline_module.run_baseline("yolo11", manifest_path, tmp_path / "report.json", invalid_view_path)


def test_run_baseline_persists_blocked_yolo_report(repo_root: Path, tmp_path: Path, monkeypatch) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json"
    dataset_view_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_detector_view.json"
    report_path = tmp_path / "blocked-yolo-report.json"
    monkeypatch.setitem(
        run_baseline_module.RUNNER_FACTORIES,
        "yolo11",
        lambda dataset_view=None, **_: _BlockedYoloRunner(dataset_view),
    )

    report = run_baseline_module.run_baseline("yolo11", manifest_path, report_path, dataset_view_path)
    saved = read_json(report_path)

    assert report["status"] == "blocked"
    assert saved["execution_config"]["model_variant"] == "yolo11n"
    assert "hardware budget" in saved["notes"]
