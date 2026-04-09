from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.config.phase1_settings import SourceConfig
from src.data.ingestion.models import RawAnnotation, RawAsset


def load_open_images_source(source: SourceConfig, base_dir: Path) -> list[RawAsset]:
    source_path = source.resolved_path(base_dir)
    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assets: list[RawAsset] = []
    for record in payload.get("records", []):
        content_hash = hashlib.sha256(f"{source.source_id}:{record['file_name']}".encode("utf-8")).hexdigest()
        assets.append(
            RawAsset(
                asset_id=f"{source.source_id}-{record['image_id']}",
                source_id=source.source_id,
                original_identifier=str(record["image_id"]),
                relative_path=record["file_name"],
                width=int(record["width"]),
                height=int(record["height"]),
                content_hash=content_hash,
                preferred_split=source.preferred_split,
                annotations=[
                    RawAnnotation(
                        source_label=item["label"],
                        bbox_xyxy=[float(value) for value in item["bbox_xyxy"]],
                    )
                    for item in record.get("annotations", [])
                ],
            )
        )
    return assets

