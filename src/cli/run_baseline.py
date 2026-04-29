from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import typer
import yaml

from src.data.manifests.models import BenchmarkManifest
from src.data.views.models import DetectorDatasetView
from src.data.views.validator import validate_detector_dataset_view
from src.evaluation.metrics.service import compute_detection_metrics
from src.evaluation.reporting.artifacts import export_failure_examples
from src.evaluation.reporting.run_report import build_run_report, save_run_report
from src.models.florence2.runner import Florence2Runner
from src.models.grounding_dino.runner import GroundingDINORunner
from src.models.yolo11.runner import YOLO11Runner
from src.utils.paths import artifact_path, read_json


RUNNER_FACTORIES = {
    "grounding_dino": lambda **_: GroundingDINORunner(),
    "florence2": lambda **_: Florence2Runner(),
    "yolo11": lambda dataset_view=None, **_: YOLO11Runner(dataset_view=dataset_view),
}
RUN_MODE_BY_MODEL = {
    "grounding_dino": "zero_shot",
    "florence2": "trainable",
    "yolo11": "trainable",
}


def _resolve_view_path(path_value: str, metadata_path: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((metadata_path.parent / path).resolve())


def _load_detector_view(dataset_view_path: str | Path) -> DetectorDatasetView:
    metadata_path = Path(dataset_view_path).resolve()
    payload = read_json(metadata_path)
    validate_detector_dataset_view(payload)
    payload["dataset_root"] = _resolve_view_path(payload["dataset_root"], metadata_path)
    payload["dataset_yaml_path"] = _resolve_view_path(payload["dataset_yaml_path"], metadata_path)
    payload["split_exports"] = {
        split_name: {
            **split_payload,
            "images_dir": _resolve_view_path(split_payload["images_dir"], metadata_path),
            "labels_dir": _resolve_view_path(split_payload["labels_dir"], metadata_path),
        }
        for split_name, split_payload in payload["split_exports"].items()
    }
    return DetectorDatasetView.model_validate(payload)


def _validate_yolo_dataset_view(manifest: BenchmarkManifest, dataset_view: DetectorDatasetView) -> None:
    if dataset_view.manifest_id != manifest.manifest_id:
        raise ValueError("Detector view manifest_id does not match the requested benchmark manifest.")

    expected_class_order = [item.class_id for item in manifest.classes if item.status == "active"]
    if dataset_view.class_order != expected_class_order:
        raise ValueError("Detector view class_order must match the active manifest class order.")

    expected_counts = {
        "train": int(manifest.asset_counts.get("train", 0)),
        "val": int(manifest.asset_counts.get("val", 0)),
        "test_real_heldout": int(manifest.asset_counts.get("test_real_heldout", 0)),
    }
    for split_name, expected_count in expected_counts.items():
        actual_count = dataset_view.split_exports[split_name].asset_count
        if actual_count != expected_count:
            raise ValueError(
                f"Detector view split count mismatch for {split_name}: expected {expected_count}, got {actual_count}."
            )

    dataset_yaml_path = Path(dataset_view.dataset_yaml_path)
    if not dataset_yaml_path.exists():
        raise ValueError(f"Detector view dataset YAML is missing: {dataset_yaml_path}")
    dataset_yaml = yaml.safe_load(dataset_yaml_path.read_text(encoding="utf-8")) or {}
    if "test_real_heldout" in str(dataset_yaml.get("train", "")) or "test_real_heldout" in str(
        dataset_yaml.get("val", "")
    ):
        raise ValueError("Held-out test data cannot be repurposed as train or val input for YOLO11.")
    if "test_real_heldout" not in str(dataset_yaml.get("test", "")):
        raise ValueError("Detector view dataset YAML must preserve the held-out test split.")
    names = dataset_yaml.get("names", {})
    if len(names) != len(dataset_view.class_order):
        raise ValueError("Detector view dataset YAML names must match the canonical class order.")


def run_baseline(
    model: str,
    manifest_path: str | Path,
    report_path: str | Path,
    dataset_view_path: str | Path | None = None,
) -> dict:
    manifest = BenchmarkManifest.model_validate(read_json(manifest_path))
    if model not in RUNNER_FACTORIES:
        raise ValueError(f"Unsupported model family: {model}")
    if model != "yolo11" and dataset_view_path is not None:
        raise ValueError("--dataset-view is only valid when --model yolo11 is selected.")

    dataset_view = None
    if model == "yolo11":
        if dataset_view_path is None:
            raise ValueError("--dataset-view is required when --model yolo11 is selected.")
        dataset_view = _load_detector_view(dataset_view_path)
        _validate_yolo_dataset_view(manifest, dataset_view)

    runner = RUNNER_FACTORIES[model](dataset_view=dataset_view)
    started = time.perf_counter()
    result = runner.run(manifest)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 4)

    if result.status != "completed":
        report = build_run_report(
            run_id=f"{model}-{int(time.time())}",
            model_family=model,
            run_mode=RUN_MODE_BY_MODEL[model],
            manifest=manifest,
            status=result.status,
            execution_config=result.execution_config,
            metrics={"latency_ms_per_image": elapsed_ms, "peak_vram_mb": 0},
            failures=[],
            notes=result.notes,
        )
        save_run_report(report, report_path)
        return report

    metrics, failures = compute_detection_metrics(manifest, result.predictions)
    metrics["latency_ms_per_image"] = elapsed_ms / max(len(manifest.assets), 1)
    metrics["peak_vram_mb"] = 0
    run_id = f"{model}-{int(time.time())}"
    failures = [
        failure.model_copy(update={"run_id": run_id})
        for failure in export_failure_examples(failures, artifact_path("failure_examples", run_id))
    ]
    report = build_run_report(
        run_id=run_id,
        model_family=model,
        run_mode=RUN_MODE_BY_MODEL[model],
        manifest=manifest,
        status="completed",
        execution_config=result.execution_config,
        metrics=metrics,
        failures=failures,
        notes=result.notes,
    )
    save_run_report(report, report_path)
    return report


def run_baseline_command(
    model: str = typer.Option(..., help="grounding_dino, florence2, or yolo11"),
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    report: Path = typer.Option(...),
    dataset_view: Optional[Path] = typer.Option(None, exists=True, file_okay=True, dir_okay=False),
) -> None:
    payload = run_baseline(model, manifest, report, dataset_view)
    typer.echo(f"Wrote baseline report {payload['run_id']} to {report}")
