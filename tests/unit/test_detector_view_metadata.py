from __future__ import annotations

from pathlib import Path

import yaml

from src.data.manifests.models import BenchmarkManifest
from src.data.views.exporter import export_detector_dataset_view
from src.utils.paths import read_json


def test_exported_detector_view_preserves_provenance_and_split_isolation(repo_root: Path, tmp_path: Path) -> None:
    manifest_path = repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json"
    manifest = BenchmarkManifest.model_validate(read_json(manifest_path))
    output_dir = tmp_path / "yolo11"
    metadata_path = tmp_path / "view.json"

    payload = export_detector_dataset_view(
        manifest,
        output_dir,
        metadata_path,
        manifest_base_dir=manifest_path.parent,
    )
    provenance = read_json(output_dir / "provenance-index.json")
    dataset_yaml = yaml.safe_load((output_dir / "dataset.yaml").read_text(encoding="utf-8"))

    assert payload["class_order"] == ["mug", "book"]
    assert provenance[0]["source_id"] == "coco2017"
    assert "test_real_heldout/images" == dataset_yaml["test"]
    assert (output_dir / "train" / "images" / "coco2017__asset-train.jpg").exists()
