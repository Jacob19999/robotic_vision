from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from src.config.phase1_settings import SourceConfig
from src.data.ingestion.models import RawAnnotation, RawAsset


def load_coco_source(source: SourceConfig, base_dir: Path) -> list[RawAsset]:
    source_path = source.resolved_path(base_dir)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = {item["id"]: item["name"] for item in payload.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    assets: list[RawAsset] = []
    for image in payload.get("images", []):
        file_name = image["file_name"]
        raw_annotations = [
            RawAnnotation(
                source_label=categories[item["category_id"]],
                bbox_xyxy=[
                    float(item["bbox"][0]),
                    float(item["bbox"][1]),
                    float(item["bbox"][0] + item["bbox"][2]),
                    float(item["bbox"][1] + item["bbox"][3]),
                ],
            )
            for item in annotations_by_image.get(image["id"], [])
            if item["category_id"] in categories
        ]
        content_hash = hashlib.sha256(f"{source.source_id}:{file_name}".encode("utf-8")).hexdigest()
        assets.append(
            RawAsset(
                asset_id=f"{source.source_id}-{image['id']}",
                source_id=source.source_id,
                original_identifier=str(image["id"]),
                relative_path=file_name,
                width=int(image["width"]),
                height=int(image["height"]),
                content_hash=content_hash,
                preferred_split=source.preferred_split,
                annotations=raw_annotations,
            )
        )
    return assets

