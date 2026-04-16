from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.paths import get_project_root


class OntologyClassConfig(BaseModel):
    class_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    status: Literal["active", "deferred"] = "active"

    @field_validator("aliases")
    @classmethod
    def normalize_aliases(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]


class SourceConfig(BaseModel):
    source_id: str
    name: str
    source_type: Literal["public_curated", "community_vetted", "in_house"]
    format: Literal["coco", "open_images"]
    path: str
    preferred_split: Literal["train", "val", "test_real_heldout"] | None = None

    def resolved_path(self, base_dir: Path) -> Path:
        candidate = Path(self.path)
        return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


class SplitConfig(BaseModel):
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42

    @model_validator(mode="after")
    def validate_ratios(self) -> "SplitConfig":
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if total != 1.0:
            raise ValueError("Split ratios must sum to 1.0")
        return self


class BaselineRuntimeConfig(BaseModel):
    model_variant: str
    fallback_variant: str | None = None
    resolution: int = 640
    precision_mode: str = "fp16"
    batch_size: int = 1
    seed: int | None = 42


class Phase1BaselineSettings(BaseModel):
    grounding_dino: BaselineRuntimeConfig = Field(
        default_factory=lambda: BaselineRuntimeConfig(
            model_variant="grounding-dino-base",
            resolution=640,
            precision_mode="fp16",
            batch_size=1,
            seed=42,
        )
    )
    florence2: BaselineRuntimeConfig = Field(
        default_factory=lambda: BaselineRuntimeConfig(
            model_variant="florence2-base",
            resolution=640,
            precision_mode="fp16",
            batch_size=1,
            seed=42,
        )
    )
    yolo11: BaselineRuntimeConfig = Field(
        default_factory=lambda: BaselineRuntimeConfig(
            model_variant="yolo11s",
            fallback_variant="yolo11n",
            resolution=640,
            precision_mode="fp16",
            batch_size=1,
            seed=42,
        )
    )


class Phase1Config(BaseModel):
    manifest_id: str
    ontology_version: str
    ontology: list[OntologyClassConfig]
    sources: list[SourceConfig]
    splits: SplitConfig = Field(default_factory=SplitConfig)
    fail_on_unmapped: bool = True
    baselines: Phase1BaselineSettings = Field(default_factory=Phase1BaselineSettings)


def load_phase1_config(config_path: str | Path) -> tuple[Phase1Config, Path]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Phase1Config.model_validate(payload), path.parent


def default_phase1_config_path() -> Path:
    return get_project_root() / "config" / "phase1.yaml"


def load_phase1_baseline_settings(config_path: str | Path | None = None) -> Phase1BaselineSettings:
    path = Path(config_path).resolve() if config_path is not None else default_phase1_config_path()
    if not path.exists():
        return Phase1BaselineSettings()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Phase1BaselineSettings.model_validate(payload.get("baselines", {}))
