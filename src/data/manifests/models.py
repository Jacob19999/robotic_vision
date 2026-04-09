from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.data.ontology.models import HouseholdObjectClass


SplitName = Literal["train", "val", "test_real_heldout"]


class DatasetAnnotation(BaseModel):
    annotation_id: str
    class_id: str
    source_label: str
    bbox_xyxy: list[float]
    is_ignored: bool = False

    @field_validator("bbox_xyxy")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values")
        return value


class DatasetAsset(BaseModel):
    asset_id: str
    source_id: str
    original_identifier: str
    relative_path: str
    width: int
    height: int
    split_name: SplitName | None = None
    content_hash: str
    review_status: Literal["accepted", "rejected", "held_for_review"] = "accepted"
    annotations: list[DatasetAnnotation] = Field(default_factory=list)
    preferred_split: SplitName | None = None


class SplitDefinition(BaseModel):
    split_name: SplitName
    purpose: str
    selection_rules: str
    locked_at: str
    version: str


class BenchmarkManifest(BaseModel):
    manifest_id: str
    ontology_version: str
    source_ids: list[str]
    split_versions: dict[str, str]
    asset_counts: dict[str, int | dict[str, int]]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    classes: list[HouseholdObjectClass] = Field(default_factory=list)
    assets: list[DatasetAsset] = Field(default_factory=list)


class FailureExample(BaseModel):
    failure_id: str
    run_id: str
    asset_id: str
    failure_type: str
    expected_class_id: str | None = None
    predicted_class_id: str | None = None
    artifact_path: str | None = None
    notes: str | None = None

