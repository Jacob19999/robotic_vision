from __future__ import annotations

from pathlib import Path

from src.config.phase1_settings import SourceConfig
from src.data.ingestion.coco import load_coco_source
from src.data.ingestion.models import RawAsset
from src.data.ingestion.open_images import load_open_images_source


SOURCE_LOADERS = {
    "coco": load_coco_source,
    "open_images": load_open_images_source,
}


def load_source_assets(source: SourceConfig, base_dir: Path) -> list[RawAsset]:
    loader = SOURCE_LOADERS[source.format]
    return loader(source, base_dir)

