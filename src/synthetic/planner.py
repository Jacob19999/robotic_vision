from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.data.manifests.models import BenchmarkManifest
from src.synthetic.models import Phase2SyntheticConfig, SyntheticGenerationPlan
from src.utils.paths import read_json, write_json


DEFAULT_NVIDIA_ASSET_SOURCE = "https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html"


def load_phase2_synthetic_config(config_path: str | Path) -> Phase2SyntheticConfig:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return Phase2SyntheticConfig.model_validate(payload)


def _active_manifest_classes(manifest: BenchmarkManifest) -> list[str]:
    active = [item.class_id for item in manifest.classes if item.status == "active"]
    if active:
        return active
    by_class = manifest.asset_counts.get("by_class", {})
    return sorted(str(class_id) for class_id in by_class)


def _failure_types_from_summary(summary_payload: dict[str, Any] | None, limit: int) -> list[str]:
    if not summary_payload:
        return []

    top_failures = summary_payload.get("top_failures")
    if isinstance(top_failures, dict):
        return [str(item[0]) for item in sorted(top_failures.items(), key=lambda pair: pair[1], reverse=True)[:limit]]
    if isinstance(top_failures, list):
        values: list[str] = []
        for item in top_failures[:limit]:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                values.append(str(item.get("failure_type") or item.get("category") or item.get("name")))
        return [value for value in values if value and value != "None"]

    issues = summary_payload.get("today_issues_and_blockers")
    if isinstance(issues, list):
        normalized: list[str] = []
        for issue in issues:
            text = str(issue).lower()
            if "book" in text or "class" in text:
                normalized.append("class imbalance")
            if "localization" in text or "ap" in text:
                normalized.append("scale or distance issue")
        return sorted(set(normalized))[:limit]
    return []


def _target_classes_from_summary(
    manifest: BenchmarkManifest,
    summary_payload: dict[str, Any] | None,
    fallback_limit: int,
) -> list[str]:
    active_classes = _active_manifest_classes(manifest)
    if not summary_payload:
        return active_classes

    per_class_ap = (
        summary_payload.get("key_results", {})
        .get("metrics", {})
        .get("per_class_ap50_95")
    )
    if isinstance(per_class_ap, dict) and per_class_ap:
        ranked = sorted(
            ((str(class_id), float(value)) for class_id, value in per_class_ap.items() if str(class_id) in active_classes),
            key=lambda item: item[1],
        )
        if ranked:
            return [class_id for class_id, _ in ranked[:fallback_limit]]

    return active_classes


def build_synthetic_generation_plan(
    manifest: BenchmarkManifest,
    config: Phase2SyntheticConfig,
    *,
    phase1_summary: dict[str, Any] | None = None,
) -> SyntheticGenerationPlan:
    target_classes = _target_classes_from_summary(manifest, phase1_summary, config.failure_focus_limit)
    target_failure_types = _failure_types_from_summary(phase1_summary, config.failure_focus_limit)

    scenes = []
    remaining_budget = config.max_total_frames
    for scene in config.scenes:
        scoped_targets = [class_id for class_id in scene.target_classes if class_id in target_classes]
        if not scoped_targets:
            scoped_targets = target_classes
        frame_count = min(scene.frame_count, remaining_budget)
        if frame_count <= 0:
            break
        scenes.append(scene.model_copy(update={"target_classes": scoped_targets, "frame_count": frame_count}))
        remaining_budget -= frame_count

    return SyntheticGenerationPlan(
        plan_id=config.plan_id,
        manifest_id=manifest.manifest_id,
        seed=config.seed,
        output_root=config.output_root,
        source_references=config.asset_packs,
        target_classes=target_classes,
        target_failure_types=target_failure_types,
        scenes=scenes,
        total_frame_budget=sum(scene.frame_count for scene in scenes),
        heldout_policy=config.heldout_policy,
        output_contract={
            "rgb": "PNG or JPEG RGB frames generated under synthetic train split only.",
            "boxes": "2D tight bounding boxes exported by Isaac Sim Replicator and converted to YOLO txt normalized xywh.",
            "dataset_view": "Synthetic detector view must use the same class_order as the Phase 1 YOLO detector view.",
            "validation": "Real validation and test splits remain unchanged for model selection and final reporting.",
        },
    )


def write_synthetic_generation_plan(
    *,
    manifest_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    phase1_summary_path: str | Path | None = None,
) -> dict:
    manifest = BenchmarkManifest.model_validate(read_json(manifest_path))
    config = load_phase2_synthetic_config(config_path)
    summary = read_json(phase1_summary_path) if phase1_summary_path is not None and Path(phase1_summary_path).exists() else None
    plan = build_synthetic_generation_plan(manifest, config, phase1_summary=summary)
    payload = plan.model_dump(mode="json")
    write_json(output_path, payload)
    return payload

