from __future__ import annotations

import shutil
from pathlib import Path

import typer
import yaml

from src.utils.paths import write_json


def _resolve_image_for_label(images_dir: Path, stem: str) -> Path:
    candidates = sorted(images_dir.glob(f"{stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"No image found for label stem '{stem}' in {images_dir}")
    return candidates[0]


def export_yolo_book_oversampled_view(
    *,
    source_dataset_root: str | Path,
    output_dataset_root: str | Path,
    oversample_factor: int = 2,
) -> dict:
    if oversample_factor < 2:
        raise ValueError("oversample_factor must be >= 2.")

    source_root = Path(source_dataset_root).resolve()
    output_root = Path(output_dataset_root).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    shutil.copytree(source_root, output_root)

    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
    names = dataset_yaml.get("names", {})
    names_lookup = {int(index): str(name) for index, name in names.items()}
    book_index = next((index for index, name in names_lookup.items() if name == "book"), None)
    if book_index is None:
        raise ValueError("Could not find class 'book' in dataset.yaml names mapping.")

    train_images_dir = output_root / "train" / "images"
    train_labels_dir = output_root / "train" / "labels"
    label_files = sorted(train_labels_dir.glob("*.txt"))
    oversampled_assets = 0
    added_train_images = 0

    for label_path in label_files:
        label_text = label_path.read_text(encoding="utf-8")
        lines = [line.strip() for line in label_text.splitlines() if line.strip()]
        has_book = any(line.startswith(f"{book_index} ") for line in lines)
        if not has_book:
            continue

        image_path = _resolve_image_for_label(train_images_dir, label_path.stem)
        oversampled_assets += 1
        for replica_index in range(1, oversample_factor):
            replica_stem = f"{label_path.stem}__bookos{replica_index}"
            replica_label_path = train_labels_dir / f"{replica_stem}.txt"
            replica_image_path = train_images_dir / f"{replica_stem}{image_path.suffix}"
            shutil.copy2(label_path, replica_label_path)
            shutil.copy2(image_path, replica_image_path)
            added_train_images += 1

    summary = {
        "report_type": "yolo11_book_oversampled_dataset",
        "source_dataset_root": str(source_root),
        "output_dataset_root": str(output_root),
        "oversample_factor": oversample_factor,
        "book_class_index": book_index,
        "book_source_assets_oversampled": oversampled_assets,
        "additional_train_images": added_train_images,
    }
    write_json(output_root / "oversample-summary.json", summary)
    return summary


def export_yolo_book_oversampled_view_command(
    source_dataset_root: Path = typer.Option(Path("artifacts/datasets/yolo11"), exists=True, file_okay=False),
    output_dataset_root: Path = typer.Option(Path("artifacts/datasets/yolo11-book-oversampled")),
    oversample_factor: int = typer.Option(2),
) -> None:
    payload = export_yolo_book_oversampled_view(
        source_dataset_root=source_dataset_root,
        output_dataset_root=output_dataset_root,
        oversample_factor=oversample_factor,
    )
    typer.echo(
        "Created oversampled dataset at "
        f"{payload['output_dataset_root']} (+{payload['additional_train_images']} train images)."
    )
