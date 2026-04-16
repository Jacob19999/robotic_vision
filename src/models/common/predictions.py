from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExecutionConfig(BaseModel):
    model_variant: str
    resolution: int = Field(ge=1)
    precision_mode: str
    batch_size: int = Field(ge=1)
    seed: int | None = None
    dataset_view_id: str | None = None
    checkpoint_reference: str | None = None


class PredictedAnnotation(BaseModel):
    class_id: str
    bbox_xyxy: list[float]
    score: float = 1.0


class AssetPrediction(BaseModel):
    asset_id: str
    predictions: list[PredictedAnnotation] = Field(default_factory=list)


class RunnerResult(BaseModel):
    status: Literal["completed", "blocked", "failed"]
    execution_config: ExecutionConfig
    predictions: list[AssetPrediction] = Field(default_factory=list)
    notes: str | None = None
