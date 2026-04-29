from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from src.data.manifests.models import BenchmarkManifest
from src.data.mixed.yolo import export_mixed_yolo_experiments
from src.data.views.exporter import export_detector_dataset_view
from src.data.views.models import DetectorDatasetView
from src.utils.paths import read_json


def _export_real_view(repo_root: Path, tmp_path: Path) -> DetectorDatasetView:
    manifest = BenchmarkManifest.model_validate(
        read_json(repo_root / "tests" / "fixtures" / "phase1_yolo" / "sample_benchmark_manifest.json")
    )
    payload = export_detector_dataset_view(
        manifest,
        tmp_path / "real-yolo",
        tmp_path / "real-view.json",
        manifest_base_dir=repo_root / "tests" / "fixtures" / "phase1_yolo",
    )
    return DetectorDatasetView.model_validate(payload)


def _write_synthetic_dataset(repo_root: Path, root: Path) -> None:
    train_images = root / "train" / "images"
    train_labels = root / "train" / "labels"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    shutil.copy2(repo_root / "tests" / "fixtures" / "phase1_yolo" / "source" / "book-val.jpg", train_images / "syn-book.jpg")
    (train_labels / "syn-book.txt").write_text("1 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
    (root / "dataset.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "train/images",
                "val": "val/images",
                "test": "test_real_heldout/images",
                "names": {0: "mug", 1: "book"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_export_mixed_yolo_experiments_preserves_real_eval_splits(repo_root: Path, tmp_path: Path) -> None:
    real_view = _export_real_view(repo_root, tmp_path)
    synthetic_root = tmp_path / "synthetic-yolo"
    _write_synthetic_dataset(repo_root, synthetic_root)

    payload = export_mixed_yolo_experiments(
        real_view=real_view,
        synthetic_dataset_root=synthetic_root,
        output_root=tmp_path / "phase3",
        target_classes=["book"],
        oversample_factor=3,
    )

    experiments = {item["experiment_name"]: item for item in payload["experiments"]}
    assert set(experiments) == {"real_only", "synthetic_only", "mixed", "targeted_synthetic_oversampling"}
    assert experiments["synthetic_only"]["train_image_count"] == 1
    assert experiments["synthetic_only"]["val_image_count"] == 1
    assert experiments["synthetic_only"]["test_real_heldout_image_count"] == 1
    assert experiments["mixed"]["train_image_count"] == 2
    assert experiments["targeted_synthetic_oversampling"]["synthetic_train_image_count"] == 3
    assert payload["selection_rule"] == "Choose checkpoints using real validation data only."

    targeted_yaml = yaml.safe_load(
        Path(experiments["targeted_synthetic_oversampling"]["dataset_yaml_path"]).read_text(encoding="utf-8")
    )
    assert targeted_yaml["test"] == "test_real_heldout/images"
