from __future__ import annotations

from pathlib import Path

from src.data.manifests.models import BenchmarkManifest
from src.data.views.exporter import export_detector_dataset_view
from src.data.views.models import DetectorDatasetView
from src.models.yolo11.runner import YOLO11Runner
from src.utils.paths import read_json


class _FakeBoxes:
    def __init__(self, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.xyxy = xyxy
        self.cls = cls
        self.conf = conf


class _FakeResult:
    def __init__(self, xyxy, cls, conf) -> None:  # noqa: ANN001
        self.boxes = _FakeBoxes(xyxy, cls, conf)


class _FakeYOLOModel:
    def __init__(self, predictions_by_name: dict[str, list[float]]) -> None:
        self.predictions_by_name = predictions_by_name

    def predict(self, source: str, **kwargs):  # noqa: ANN003
        bbox = self.predictions_by_name[Path(source).name]
        return [_FakeResult([bbox], [0], [0.98])]


def _load_manifest(repo_root: Path) -> BenchmarkManifest:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json"
    return BenchmarkManifest.model_validate(read_json(manifest_path))


def _export_view(manifest: BenchmarkManifest, repo_root: Path, tmp_path: Path) -> DetectorDatasetView:
    payload = export_detector_dataset_view(
        manifest,
        tmp_path / "yolo11",
        tmp_path / "view.json",
        manifest_base_dir=repo_root / "tests" / "fixtures" / "phase1_yolo",
    )
    return DetectorDatasetView.model_validate(payload)


def test_yolo11_runner_smoke_predicts_with_primary_variant(repo_root: Path, tmp_path: Path) -> None:
    manifest = _load_manifest(repo_root)
    dataset_view = _export_view(manifest, repo_root, tmp_path)
    predictions = {
        "coco2017__asset-train.jpg": [10.0, 12.0, 90.0, 112.0],
        "coco2017__asset-val.jpg": [30.0, 40.0, 150.0, 160.0],
        "open_images_v7__asset-test.jpg": [5.0, 10.0, 55.0, 80.0],
    }
    runner = YOLO11Runner(dataset_view, model_loader=lambda model_variant: _FakeYOLOModel(predictions))

    result = runner.run(manifest)

    assert result.status == "completed"
    assert result.execution_config.model_variant == "yolo11s"
    assert result.execution_config.dataset_view_id == dataset_view.view_id
    assert len(result.predictions) == 3


def test_yolo11_runner_falls_back_to_smaller_variant(repo_root: Path, tmp_path: Path) -> None:
    manifest = _load_manifest(repo_root)
    dataset_view = _export_view(manifest, repo_root, tmp_path)
    predictions = {
        "coco2017__asset-train.jpg": [10.0, 12.0, 90.0, 112.0],
        "coco2017__asset-val.jpg": [30.0, 40.0, 150.0, 160.0],
        "open_images_v7__asset-test.jpg": [5.0, 10.0, 55.0, 80.0],
    }

    def loader(model_variant: str):
        if model_variant == "yolo11s":
            raise RuntimeError("CUDA out of memory while loading yolo11s")
        return _FakeYOLOModel(predictions)

    runner = YOLO11Runner(dataset_view, model_loader=loader)
    result = runner.run(manifest)

    assert result.status == "completed"
    assert result.execution_config.model_variant == "yolo11n"
    assert "fell back from yolo11s to yolo11n" in (result.notes or "")
