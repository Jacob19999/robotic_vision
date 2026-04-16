from __future__ import annotations

from pathlib import Path

from src.data.manifests.models import BenchmarkManifest, DatasetAnnotation, DatasetAsset
from src.data.views.models import YOLOClassMapping


def build_yolo_class_mapping(manifest: BenchmarkManifest) -> YOLOClassMapping:
    class_order = [item.class_id for item in manifest.classes if item.status == "active"]
    if not class_order:
        class_order = sorted(manifest.asset_counts.get("by_class", {}))
    return YOLOClassMapping(
        mapping_id=f"{manifest.manifest_id}-yolo-class-map",
        manifest_id=manifest.manifest_id,
        class_order=class_order,
        index_by_class_id={class_id: index for index, class_id in enumerate(class_order)},
    )


def xyxy_to_normalized_xywh(
    bbox_xyxy: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = bbox_xyxy
    x1 = min(max(float(x1), 0.0), float(image_width))
    y1 = min(max(float(y1), 0.0), float(image_height))
    x2 = min(max(float(x2), 0.0), float(image_width))
    y2 = min(max(float(y2), 0.0), float(image_height))
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox_xyxy must define a non-negative box")
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + (width / 2.0)
    y_center = y1 + (height / 2.0)
    return (
        round(x_center / image_width, 6),
        round(y_center / image_height, 6),
        round(width / image_width, 6),
        round(height / image_height, 6),
    )


def annotation_to_yolo_line(
    annotation: DatasetAnnotation,
    image_width: int,
    image_height: int,
    class_mapping: YOLOClassMapping,
) -> str:
    class_index = class_mapping.index_by_class_id[annotation.class_id]
    x_center, y_center, width, height = xyxy_to_normalized_xywh(
        annotation.bbox_xyxy,
        image_width,
        image_height,
    )
    return f"{class_index} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def asset_to_yolo_lines(asset: DatasetAsset, class_mapping: YOLOClassMapping) -> list[str]:
    lines: list[str] = []
    for annotation in asset.annotations:
        if annotation.is_ignored:
            continue
        lines.append(
            annotation_to_yolo_line(
                annotation,
                asset.width,
                asset.height,
                class_mapping,
            )
        )
    return lines


def exported_asset_name(asset: DatasetAsset) -> str:
    suffix = Path(asset.relative_path).suffix or ".jpg"
    return f"{asset.source_id}__{asset.asset_id}{suffix}"
