from __future__ import annotations

from pathlib import Path

import typer

from src.config.phase1_settings import load_phase1_baseline_settings
from src.data.manifests.models import BenchmarkManifest
from src.data.views.exporter import export_detector_dataset_view
from src.utils.paths import read_json


def export_yolo_view(
    manifest_path: str | Path,
    output_dir: str | Path,
    metadata_path: str | Path,
    *,
    model_variant: str | None = None,
) -> dict:
    manifest = BenchmarkManifest.model_validate(read_json(manifest_path))
    baseline_settings = load_phase1_baseline_settings().yolo11
    return export_detector_dataset_view(
        manifest,
        output_dir,
        metadata_path,
        manifest_base_dir=Path(manifest_path).resolve().parent,
        model_variant=model_variant or baseline_settings.model_variant,
    )


def export_yolo_view_command(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
    metadata: Path = typer.Option(...),
) -> None:
    payload = export_yolo_view(manifest, output_dir, metadata)
    typer.echo(f"Wrote detector view {payload['view_id']} to {metadata}")
