from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class YOLOClassMapping(BaseModel):
    mapping_id: str
    manifest_id: str
    class_order: list[str] = Field(min_length=1)
    index_by_class_id: dict[str, int]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def validate_mapping(self) -> "YOLOClassMapping":
        if len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order must contain unique canonical class ids")
        expected = {class_id: index for index, class_id in enumerate(self.class_order)}
        if self.index_by_class_id != expected:
            raise ValueError("index_by_class_id must match the contiguous zero-based class_order")
        return self


class DetectorSplitExport(BaseModel):
    images_dir: str
    labels_dir: str
    asset_count: int = Field(ge=0)


class DetectorDatasetView(BaseModel):
    view_id: str
    manifest_id: str
    model_family: str
    model_variant: str
    dataset_root: str
    dataset_yaml_path: str
    annotation_format: Literal["yolo_txt_normalized_xywh"] = "yolo_txt_normalized_xywh"
    class_order: list[str] = Field(min_length=1)
    split_exports: dict[Literal["train", "val", "test_real_heldout"], DetectorSplitExport]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @model_validator(mode="after")
    def validate_view(self) -> "DetectorDatasetView":
        if len(set(self.class_order)) != len(self.class_order):
            raise ValueError("class_order must contain unique canonical class ids")
        required_splits = {"train", "val", "test_real_heldout"}
        if set(self.split_exports) != required_splits:
            raise ValueError("split_exports must define train, val, and test_real_heldout")
        test_dir = self.split_exports["test_real_heldout"].images_dir
        for split_name in ("train", "val"):
            if self.split_exports[split_name].images_dir == test_dir:
                raise ValueError("test_real_heldout export must not reuse train or val image directories")
        return self
