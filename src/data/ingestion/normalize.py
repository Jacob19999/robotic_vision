from __future__ import annotations

from src.data.ingestion.models import RawAsset
from src.data.manifests.models import DatasetAnnotation, DatasetAsset
from src.data.ontology.registry import OntologyRegistry


def normalize_assets(
    raw_assets: list[RawAsset],
    registry: OntologyRegistry,
    fail_on_unmapped: bool = True,
) -> list[DatasetAsset]:
    normalized: list[DatasetAsset] = []
    for asset in raw_assets:
        annotations: list[DatasetAnnotation] = []
        for index, annotation in enumerate(asset.annotations):
            class_id = registry.map_label(annotation.source_label)
            if class_id is None:
                if fail_on_unmapped:
                    raise ValueError(f"Unmapped label '{annotation.source_label}' in asset {asset.asset_id}")
                continue
            annotations.append(
                DatasetAnnotation(
                    annotation_id=f"{asset.asset_id}-ann-{index}",
                    class_id=class_id,
                    source_label=annotation.source_label,
                    bbox_xyxy=annotation.bbox_xyxy,
                    is_ignored=annotation.is_ignored,
                )
            )
        if not annotations:
            continue
        normalized.append(
            DatasetAsset(
                asset_id=asset.asset_id,
                source_id=asset.source_id,
                original_identifier=asset.original_identifier,
                relative_path=asset.relative_path,
                width=asset.width,
                height=asset.height,
                content_hash=asset.content_hash,
                preferred_split=asset.preferred_split,
                annotations=annotations,
            )
        )
    return normalized

