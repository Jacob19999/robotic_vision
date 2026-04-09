from __future__ import annotations

import hashlib

from src.config.phase1_settings import SplitConfig
from src.data.manifests.models import DatasetAsset, SplitDefinition


def _hash_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def assign_splits(assets: list[DatasetAsset], split_config: SplitConfig) -> list[DatasetAsset]:
    assigned: list[DatasetAsset] = []
    for asset in assets:
        split_name = asset.preferred_split or asset.split_name
        if split_name is None:
            fraction = _hash_fraction(asset.content_hash or asset.asset_id, split_config.seed)
            if fraction < split_config.train_ratio:
                split_name = "train"
            elif fraction < split_config.train_ratio + split_config.val_ratio:
                split_name = "val"
            else:
                split_name = "test_real_heldout"
        assigned.append(asset.model_copy(update={"split_name": split_name}))
    assert_no_cross_split_duplicates(assigned)
    return assigned


def assert_no_cross_split_duplicates(assets: list[DatasetAsset]) -> None:
    seen: dict[str, str] = {}
    for asset in assets:
        existing = seen.get(asset.content_hash)
        if existing and existing != asset.split_name:
            raise ValueError(
                f"Duplicate content hash {asset.content_hash} appears in both {existing} and {asset.split_name}"
            )
        seen[asset.content_hash] = asset.split_name or "unknown"


def build_split_definitions(seed: int) -> dict[str, SplitDefinition]:
    locked_at = "frozen"
    return {
        "train": SplitDefinition(
            split_name="train",
            purpose="Model fitting",
            selection_rules=f"Deterministic hash split with seed {seed}",
            locked_at=locked_at,
            version="phase1-train-v1",
        ),
        "val": SplitDefinition(
            split_name="val",
            purpose="Model comparison and tuning",
            selection_rules=f"Deterministic hash split with seed {seed}",
            locked_at=locked_at,
            version="phase1-val-v1",
        ),
        "test_real_heldout": SplitDefinition(
            split_name="test_real_heldout",
            purpose="Frozen final evaluation set",
            selection_rules=f"Deterministic hash split with seed {seed}",
            locked_at=locked_at,
            version="phase1-test-v1",
        ),
    }

