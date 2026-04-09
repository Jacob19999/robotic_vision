from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HouseholdObjectClass(BaseModel):
    class_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    status: Literal["active", "deferred"] = "active"


class DatasetSource(BaseModel):
    source_id: str
    name: str
    source_type: Literal["public_curated", "community_vetted", "in_house"]
    license_reference: str | None = None
    priority_order: int = 0
    notes: str | None = None

