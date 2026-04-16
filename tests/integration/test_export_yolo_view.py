from __future__ import annotations

from pathlib import Path

import yaml

from src.cli.export_yolo_view import export_yolo_view
from src.utils.paths import read_json


def test_export_yolo_view_writes_metadata_and_labels(repo_root: Path, tmp_path: Path) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json"
    output_dir = tmp_path / "yolo11"
    metadata_path = tmp_path / "view.json"

    payload = export_yolo_view(manifest_path, output_dir, metadata_path)
    saved = read_json(metadata_path)
    dataset_yaml = yaml.safe_load((output_dir / "dataset.yaml").read_text(encoding="utf-8"))

    assert payload["view_id"] == "phase1-yolo-sample-yolo11-view"
    assert saved["split_exports"]["test_real_heldout"]["asset_count"] == 1
    assert dataset_yaml["test"] == "test_real_heldout/images"
    assert (output_dir / "train" / "labels" / "coco2017__asset-train.txt").exists()
    assert (output_dir / "test_real_heldout" / "images" / "open_images_v7__asset-test.jpg").exists()
