from __future__ import annotations

from dataclasses import dataclass

from src.config.phase1_settings import OntologyClassConfig
from src.data.ontology.models import HouseholdObjectClass


@dataclass(slots=True)
class OntologyRegistry:
    classes: dict[str, HouseholdObjectClass]
    label_map: dict[str, str]

    @classmethod
    def from_config(cls, classes: list[OntologyClassConfig]) -> "OntologyRegistry":
        registry: dict[str, HouseholdObjectClass] = {}
        label_map: dict[str, str] = {}
        for item in classes:
            model = HouseholdObjectClass(
                class_id=item.class_id,
                canonical_name=item.canonical_name,
                aliases=item.aliases,
                status=item.status,
            )
            registry[model.class_id] = model
            label_map[model.canonical_name.casefold()] = model.class_id
            for alias in model.aliases:
                label_map[alias.casefold()] = model.class_id
        return cls(classes=registry, label_map=label_map)

    def map_label(self, label: str) -> str | None:
        return self.label_map.get(label.casefold())

    def active_classes(self) -> list[HouseholdObjectClass]:
        return [item for item in self.classes.values() if item.status == "active"]

