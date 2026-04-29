from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AssetPackConfig(BaseModel):
    pack_id: str
    provider: Literal["nvidia"] = "nvidia"
    name: str
    source_url: str
    categories: list[str] = Field(default_factory=list)
    local_path_hint: str | None = None
    size_gb: float | None = None
    required: bool = False
    notes: str | None = None


class SceneArchetypeConfig(BaseModel):
    scene_id: str
    description: str
    target_classes: list[str] = Field(default_factory=list)
    frame_count: int = Field(ge=1, default=100)
    resolution_width: int = Field(ge=1, default=640)
    resolution_height: int = Field(ge=1, default=480)
    asset_pack_ids: list[str] = Field(default_factory=list)
    base_scene_usd: str | None = None
    object_usd_paths: dict[str, list[str]] = Field(default_factory=dict)
    distractor_usd_paths: list[str] = Field(default_factory=list)
    camera_distance_range_m: tuple[float, float] = (0.75, 1.8)
    camera_height_range_m: tuple[float, float] = (0.8, 1.6)
    clutter_count_range: tuple[int, int] = (1, 5)
    lighting_intensity_range: tuple[float, float] = (300.0, 1200.0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "SceneArchetypeConfig":
        numeric_ranges = (
            self.camera_distance_range_m,
            self.camera_height_range_m,
            self.lighting_intensity_range,
        )
        for lower, upper in numeric_ranges:
            if upper < lower:
                raise ValueError("range upper bound must be >= lower bound")
        clutter_lower, clutter_upper = self.clutter_count_range
        if clutter_lower < 0 or clutter_upper < clutter_lower:
            raise ValueError("clutter_count_range must be non-negative and ordered")
        return self


class Phase2SyntheticConfig(BaseModel):
    plan_id: str = "phase2-lightweight-synthetic"
    output_root: str = "artifacts/synthetic/phase2"
    seed: int = 42
    max_total_frames: int = Field(ge=1, default=500)
    failure_focus_limit: int = Field(ge=1, default=5)
    heldout_policy: str = "Synthetic images are train-only supplements; val and test stay real-only."
    asset_packs: list[AssetPackConfig]
    scenes: list[SceneArchetypeConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pack_references(self) -> "Phase2SyntheticConfig":
        known_pack_ids = {pack.pack_id for pack in self.asset_packs}
        for scene in self.scenes:
            unknown = [pack_id for pack_id in scene.asset_pack_ids if pack_id not in known_pack_ids]
            if unknown:
                raise ValueError(f"scene {scene.scene_id} references unknown asset packs: {unknown}")
        return self


class SyntheticGenerationPlan(BaseModel):
    plan_id: str
    manifest_id: str
    seed: int
    output_root: str
    source_references: list[AssetPackConfig]
    target_classes: list[str]
    target_failure_types: list[str]
    scenes: list[SceneArchetypeConfig]
    total_frame_budget: int
    heldout_policy: str
    output_contract: dict[str, str]
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

