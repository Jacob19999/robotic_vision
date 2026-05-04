from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from src.data.views.models import DetectorDatasetView
from src.utils.paths import read_json, write_json


def load_detector_view_for_mixing(real_view: Path | str) -> DetectorDatasetView:
    payload = read_json(Path(real_view))
    return DetectorDatasetView.model_validate(payload)


def _copy_split_tree(source_root: Path, dest_root: Path, split: str) -> None:
    for sub in ("images", "labels"):
        src = source_root / split / sub
        dst = dest_root / split / sub
        dst.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            continue
        for path in src.iterdir():
            shutil.copy2(path, dst / path.name)


def _copy_file(src: Path, dest_dir: Path, dest_name: str | None = None) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = dest_name or src.name
    shutil.copy2(src, dest_dir / name)


def _write_dataset_yaml(root: Path, names: dict[int, str]) -> Path:
    path = root / "dataset.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "path": str(root.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test_real_heldout/images",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _count_images(split_images_dir: Path) -> int:
    if not split_images_dir.is_dir():
        return 0
    return sum(1 for p in split_images_dir.iterdir() if p.is_file())


def export_mixed_yolo_experiments(
    *,
    real_view: DetectorDatasetView,
    synthetic_dataset_root: Path,
    output_root: Path,
    target_classes: list[str] | None = None,
    oversample_factor: int = 2,
) -> dict[str, Any]:
    """
    Materialize four YOLO dataset layouts for Phase 3 mixed retraining experiments.
    Evaluation val/test splits always mirror the real detector view (frozen held-out).
    """
    output_root = output_root.resolve()
    real_root = Path(real_view.dataset_root).resolve()
    if not real_view.dataset_yaml_path:
        raise ValueError("real_view.dataset_yaml_path is required")
    yaml_path = Path(real_view.dataset_yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = (real_root / yaml_path).resolve()
    names_block = yaml.safe_load(yaml_path.read_text(encoding="utf-8")).get("names", {})
    names = {int(k): str(v) for k, v in names_block.items()}

    syn_root = synthetic_dataset_root.resolve()
    syn_train_img = syn_root / "train" / "images"
    syn_train_lbl = syn_root / "train" / "labels"
    syn_image_files = sorted(syn_train_img.glob("*")) if syn_train_img.is_dir() else []
    if not syn_image_files:
        raise FileNotFoundError(f"No synthetic train images under {syn_train_img}")
    primary_syn = syn_image_files[0]
    primary_syn_lbl = syn_train_lbl / f"{primary_syn.stem}.txt"

    experiments: list[dict[str, Any]] = []
    matrix_id = "phase3-mixed-yolo-matrix"

    def add_experiment(
        name: str,
        root: Path,
        *,
        synthetic_train_image_count: int | None = None,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True)
        yaml_out = _write_dataset_yaml(root, names)
        train_n = _count_images(root / "train" / "images")
        val_n = _count_images(root / "val" / "images")
        test_n = _count_images(root / "test_real_heldout" / "images")
        experiments.append(
            {
                "experiment_name": name,
                "dataset_yaml_path": str(yaml_out),
                "train_image_count": train_n,
                "val_image_count": val_n,
                "test_real_heldout_image_count": test_n,
                "synthetic_train_image_count": synthetic_train_image_count,
            }
        )

    # real_only
    real_only = output_root / "real_only"
    if real_only.exists():
        shutil.rmtree(real_only)
    for split in ("train", "val", "test_real_heldout"):
        _copy_split_tree(real_root, real_only, split)
    add_experiment("real_only", real_only, synthetic_train_image_count=None)

    # synthetic_only: synthetic train, real val + test
    syn_only = output_root / "synthetic_only"
    if syn_only.exists():
        shutil.rmtree(syn_only)
    _copy_split_tree(syn_root, syn_only, "train")
    _copy_split_tree(real_root, syn_only, "val")
    _copy_split_tree(real_root, syn_only, "test_real_heldout")
    add_experiment("synthetic_only", syn_only, synthetic_train_image_count=None)

    # mixed: real train + synthetic train
    mixed = output_root / "mixed"
    if mixed.exists():
        shutil.rmtree(mixed)
    _copy_split_tree(real_root, mixed, "train")
    syn_dst_train_img = mixed / "train" / "images"
    syn_dst_train_lbl = mixed / "train" / "labels"
    _copy_file(primary_syn, syn_dst_train_img)
    if primary_syn_lbl.is_file():
        _copy_file(primary_syn_lbl, syn_dst_train_lbl)
    _copy_split_tree(real_root, mixed, "val")
    _copy_split_tree(real_root, mixed, "test_real_heldout")
    add_experiment("mixed", mixed, synthetic_train_image_count=None)

    # targeted synthetic oversampling for requested classes
    targets = target_classes or []
    targeted = output_root / "targeted_synthetic_oversampling"
    if targeted.exists():
        shutil.rmtree(targeted)
    _copy_split_tree(real_root, targeted, "train")
    t_train_img = targeted / "train" / "images"
    t_train_lbl = targeted / "train" / "labels"
    syn_count = 0
    if targets and "book" in targets:
        for index in range(max(1, oversample_factor)):
            stem = f"{primary_syn.stem}-dup{index}"
            _copy_file(primary_syn, t_train_img, f"{stem}{primary_syn.suffix}")
            if primary_syn_lbl.is_file():
                _copy_file(primary_syn_lbl, t_train_lbl, f"{stem}.txt")
            syn_count += 1
    else:
        _copy_file(primary_syn, t_train_img)
        if primary_syn_lbl.is_file():
            _copy_file(primary_syn_lbl, t_train_lbl)
        syn_count = 1
    _copy_split_tree(real_root, targeted, "val")
    _copy_split_tree(real_root, targeted, "test_real_heldout")
    add_experiment(
        "targeted_synthetic_oversampling",
        targeted,
        synthetic_train_image_count=syn_count,
    )

    payload: dict[str, Any] = {
        "matrix_id": matrix_id,
        "selection_rule": "Choose checkpoints using real validation data only.",
        "heldout_policy": "test_real_heldout is copied from the real detector view for every experiment.",
        "experiments": experiments,
    }
    write_json(output_root / "phase3-mixed-yolo-matrix.json", payload)
    return payload
