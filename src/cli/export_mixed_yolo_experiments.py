from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

from src.data.mixed.yolo import export_mixed_yolo_experiments, load_detector_view_for_mixing


def export_mixed_yolo_experiments_command(
    real_view: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    synthetic_dataset_root: Path = typer.Option(..., exists=True, file_okay=False, dir_okay=True),
    output_root: Path = typer.Option(Path("artifacts/datasets/phase3-yolo-mixed")),
    target_class: Optional[List[str]] = typer.Option(None, help="Class to oversample from synthetic train data."),
    oversample_factor: int = typer.Option(2),
) -> None:
    detector_view = load_detector_view_for_mixing(real_view)
    payload = export_mixed_yolo_experiments(
        real_view=detector_view,
        synthetic_dataset_root=synthetic_dataset_root,
        output_root=output_root,
        target_classes=target_class,
        oversample_factor=oversample_factor,
    )
    typer.echo(f"Wrote Phase 3 mixed retraining matrix {payload['matrix_id']} to {output_root}")
