from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictedAnnotation(BaseModel):
    class_id: str
    bbox_xyxy: list[float]
    score: float = 1.0


class AssetPrediction(BaseModel):
    asset_id: str
    predictions: list[PredictedAnnotation] = Field(default_factory=list)


class RunnerResult(BaseModel):
    status: Literal["completed", "blocked", "failed"]
    predictions: list[AssetPrediction] = Field(default_factory=list)
    notes: str | None = None

