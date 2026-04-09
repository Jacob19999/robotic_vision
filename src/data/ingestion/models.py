from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RawAnnotation(BaseModel):
    source_label: str
    bbox_xyxy: list[float]
    is_ignored: bool = False


class RawAsset(BaseModel):
    asset_id: str
    source_id: str
    original_identifier: str
    relative_path: str
    width: int
    height: int
    content_hash: str
    preferred_split: Literal["train", "val", "test_real_heldout"] | None = None
    annotations: list[RawAnnotation] = Field(default_factory=list)

