from __future__ import annotations

from collections import Counter

from src.config.phase1_settings import Phase1Config
from src.data.manifests.models import BenchmarkManifest, DatasetAsset
from src.data.ontology.registry import OntologyRegistry
from src.data.splits.service import assign_splits, build_split_definitions


def build_benchmark_manifest(
    assets: list[DatasetAsset],
    config: Phase1Config,
    registry: OntologyRegistry,
) -> BenchmarkManifest:
    assigned_assets = assign_splits(assets, config.splits)
    by_class: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()

    for asset in assigned_assets:
        split_counts[asset.split_name or "unknown"] += 1
        for annotation in asset.annotations:
            by_class[annotation.class_id] += 1

    split_definitions = build_split_definitions(config.splits.seed)
    return BenchmarkManifest(
        manifest_id=config.manifest_id,
        ontology_version=config.ontology_version,
        source_ids=[item.source_id for item in config.sources],
        split_versions={name: definition.version for name, definition in split_definitions.items()},
        asset_counts={
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test_real_heldout": split_counts["test_real_heldout"],
            "by_class": dict(by_class),
        },
        classes=registry.active_classes(),
        assets=assigned_assets,
    )

