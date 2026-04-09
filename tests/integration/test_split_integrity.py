from __future__ import annotations

import pytest

from src.config.phase1_settings import SplitConfig
from src.data.manifests.models import DatasetAsset
from src.data.splits.service import assign_splits


def _asset(asset_id: str, content_hash: str) -> DatasetAsset:
    return DatasetAsset(
        asset_id=asset_id,
        source_id="coco2017",
        original_identifier=asset_id,
        relative_path=f"images/{asset_id}.jpg",
        width=640,
        height=480,
        content_hash=content_hash,
    )


def test_assign_splits_is_deterministic() -> None:
    config = SplitConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=7)
    assets = [_asset("a1", "hash-1"), _asset("a2", "hash-2"), _asset("a3", "hash-3")]
    first = assign_splits(assets, config)
    second = assign_splits(assets, config)
    assert [item.split_name for item in first] == [item.split_name for item in second]


def test_assign_splits_rejects_cross_split_duplicates() -> None:
    config = SplitConfig()
    assets = [
        _asset("a1", "shared-hash").model_copy(update={"preferred_split": "train"}),
        _asset("a2", "shared-hash").model_copy(update={"preferred_split": "test_real_heldout"}),
    ]
    with pytest.raises(ValueError):
        assign_splits(assets, config)
