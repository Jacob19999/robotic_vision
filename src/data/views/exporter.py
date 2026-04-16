from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from src.data.manifests.models import BenchmarkManifest, DatasetAsset
from src.data.views.models import DetectorDatasetView, DetectorSplitExport
from src.data.views.validator import validate_detector_dataset_view
from src.data.views.yolo import asset_to_yolo_lines, build_yolo_class_mapping, exported_asset_name
from src.utils.paths import ensure_directory, write_json


def _resolve_asset_source(asset: DatasetAsset, manifest_base_dir: Path) -> Path:
    asset_path = Path(asset.relative_path)
    if asset_path.is_absolute():
        return asset_path

    for candidate in (
        (manifest_base_dir / asset_path).resolve(),
        asset_path.resolve(),
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve asset source for {asset.asset_id}: {asset.relative_path}")


def export_detector_dataset_view(
    manifest: BenchmarkManifest,
    output_dir: str | Path,
    metadata_path: str | Path,
    *,
    manifest_base_dir: str | Path | None = None,
    model_family: str = "yolo11",
    model_variant: str = "yolo11s",
) -> dict:
    output_root = ensure_directory(output_dir).resolve()
    metadata_destination = Path(metadata_path).resolve()
    base_dir = Path(manifest_base_dir).resolve() if manifest_base_dir is not None else Path.cwd()
    class_mapping = build_yolo_class_mapping(manifest)
    split_exports: dict[str, DetectorSplitExport] = {}
    provenance_index: list[dict[str, str | int]] = []

    for split_name in ("train", "val", "test_real_heldout"):
        split_root = ensure_directory(output_root / split_name)
        images_dir = ensure_directory(split_root / "images")
        labels_dir = ensure_directory(split_root / "labels")
        split_assets = [asset for asset in manifest.assets if asset.split_name == split_name]

        for asset in split_assets:
            exported_name = exported_asset_name(asset)
            source_path = _resolve_asset_source(asset, base_dir)
            exported_image_path = images_dir / exported_name
            exported_label_path = labels_dir / f"{Path(exported_name).stem}.txt"
            shutil.copy2(source_path, exported_image_path)
            lines = asset_to_yolo_lines(asset, class_mapping)
            exported_label_path.write_text(
                ("\n".join(lines) + ("\n" if lines else "")),
                encoding="utf-8",
            )
            provenance_index.append(
                {
                    "asset_id": asset.asset_id,
                    "source_id": asset.source_id,
                    "split_name": split_name,
                    "original_relative_path": asset.relative_path,
                    "exported_image_path": str(exported_image_path),
                    "exported_label_path": str(exported_label_path),
                }
            )

        split_exports[split_name] = DetectorSplitExport(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            asset_count=len(split_assets),
        )

    dataset_yaml_path = output_root / "dataset.yaml"
    dataset_yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root),
                "train": "train/images",
                "val": "val/images",
                "test": "test_real_heldout/images",
                "names": {index: class_id for index, class_id in enumerate(class_mapping.class_order)},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_json(output_root / "provenance-index.json", provenance_index)

    payload = DetectorDatasetView(
        view_id=f"{manifest.manifest_id}-{model_family}-view",
        manifest_id=manifest.manifest_id,
        model_family=model_family,
        model_variant=model_variant,
        dataset_root=str(output_root),
        dataset_yaml_path=str(dataset_yaml_path),
        class_order=class_mapping.class_order,
        split_exports=split_exports,
    ).model_dump(mode="json")
    validate_detector_dataset_view(payload)
    write_json(metadata_destination, payload)
    return payload
