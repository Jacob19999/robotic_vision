from __future__ import annotations

from src.data.manifests.models import BenchmarkManifest
from src.data.views.yolo import asset_to_yolo_lines, build_yolo_class_mapping, xyxy_to_normalized_xywh
from src.utils.paths import read_json


def test_xyxy_to_normalized_xywh_converts_boxes() -> None:
    assert xyxy_to_normalized_xywh([10.0, 20.0, 110.0, 120.0], 200, 200) == (0.3, 0.35, 0.5, 0.5)


def test_asset_to_yolo_lines_uses_manifest_class_mapping(repo_root) -> None:  # noqa: ANN001
    manifest = BenchmarkManifest.model_validate(
        read_json(repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json")
    )
    class_mapping = build_yolo_class_mapping(manifest)
    train_asset = next(asset for asset in manifest.assets if asset.asset_id == "asset-train")

    lines = asset_to_yolo_lines(train_asset, class_mapping)

    assert class_mapping.class_order == ["mug", "book"]
    assert lines == ["0 0.078125 0.129167 0.125000 0.208333"]
