from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config.phase1_settings import SourceConfig
from src.data.ingestion.coco import load_coco_source
from src.data.ingestion.open_images import load_open_images_source
from src.data.ingestion.source_registry import SOURCE_LOADERS, load_source_assets


def test_load_coco_source_maps_categories_and_bbox(repo_root: Path) -> None:
    path = repo_root / "tests" / "fixtures" / "phase1_benchmark" / "manifest_input.json"
    source = SourceConfig(
        source_id="coco2017",
        name="COCO sample",
        source_type="public_curated",
        format="coco",
        path=str(path),
    )
    assets = load_coco_source(source, repo_root)
    assert len(assets) == 2
    by_id = {a.original_identifier: a for a in assets}
    mug = by_id["1"]
    assert mug.relative_path == "images/mug-1.jpg"
    assert mug.annotations[0].source_label == "cup"
    assert mug.annotations[0].bbox_xyxy == [10.0, 12.0, 90.0, 112.0]


def test_load_open_images_source_reads_records(tmp_path: Path) -> None:
    payload = {
        "records": [
            {
                "image_id": "x1",
                "file_name": "a/b.jpg",
                "width": 640,
                "height": 480,
                "annotations": [{"label": "mug", "bbox_xyxy": [1.0, 2.0, 3.0, 4.0]}],
            }
        ]
    }
    json_path = tmp_path / "oi.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    source = SourceConfig(
        source_id="open_images_v7",
        name="OI slice",
        source_type="public_curated",
        format="open_images",
        path=str(json_path),
        preferred_split="val",
    )
    assets = load_open_images_source(source, tmp_path)
    assert len(assets) == 1
    a = assets[0]
    assert a.asset_id == "open_images_v7-x1"
    assert a.preferred_split == "val"
    assert a.annotations[0].source_label == "mug"
    assert a.annotations[0].bbox_xyxy == [1.0, 2.0, 3.0, 4.0]


def test_load_source_assets_dispatches_by_format(tmp_path: Path) -> None:
    payload = {"records": []}
    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    source = SourceConfig(
        source_id="open_images_v7",
        name="OI",
        source_type="public_curated",
        format="open_images",
        path=str(json_path),
    )
    assert load_source_assets(source, tmp_path) == []


def test_source_loaders_match_phase1_formats() -> None:
    """Phase 1 SourceConfig.format must stay in sync with SOURCE_LOADERS."""
    assert set(SOURCE_LOADERS.keys()) == {"coco", "open_images"}


def test_objects365_not_in_registry() -> None:
    with pytest.raises(KeyError):
        _ = SOURCE_LOADERS["objects365"]
