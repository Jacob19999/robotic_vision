from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer

from src.utils.paths import read_json, write_json


def _to_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def train_mixed_yolo11(
    *,
    matrix_path: str | Path,
    output_dir: str | Path,
    base_model: str = "yolo11s",
    epochs: int = 40,
    imgsz: int = 640,
    batch: int = 8,
    dry_run: bool = True,
) -> dict:
    matrix = read_json(matrix_path)
    report = {
        "report_type": "phase3_mixed_yolo11_training",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_id": matrix["matrix_id"],
        "base_model": base_model,
        "selection_rule": matrix["selection_rule"],
        "heldout_policy": matrix["heldout_policy"],
        "dry_run": dry_run,
        "experiments": [],
    }
    for experiment in matrix["experiments"]:
        entry = {
            "experiment_name": experiment["experiment_name"],
            "dataset_yaml": experiment["dataset_yaml_path"],
            "train_image_count": experiment["train_image_count"],
            "val_image_count": experiment["val_image_count"],
            "test_real_heldout_image_count": experiment["test_real_heldout_image_count"],
            "synthetic_train_image_count": experiment["synthetic_train_image_count"],
        }
        if dry_run:
            entry["status"] = "planned"
            report["experiments"].append(entry)
            continue

        from ultralytics import YOLO

        experiment_output_dir = Path(output_dir).resolve() / experiment["experiment_name"]
        model = YOLO(base_model if base_model.endswith(".pt") else f"{base_model}.pt")
        model.train(
            data=experiment["dataset_yaml_path"],
            project=str(experiment_output_dir),
            name="train",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=15,
            optimizer="AdamW",
            save=True,
        )
        best_checkpoint = experiment_output_dir / "train" / "weights" / "best.pt"
        tuned = YOLO(str(best_checkpoint))
        metrics = tuned.val(data=experiment["dataset_yaml_path"], split="test")
        entry.update(
            {
                "status": "completed",
                "best_checkpoint": str(best_checkpoint),
                "test_real_heldout_metrics": {
                    "precision": _to_float(metrics.box.mp),
                    "recall": _to_float(metrics.box.mr),
                    "mAP50": _to_float(metrics.box.map50),
                    "mAP50_95": _to_float(metrics.box.map),
                },
            }
        )
        report["experiments"].append(entry)
    return report


def train_mixed_yolo11_command(
    matrix: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(Path("runs/detect/artifacts/phase3-mixed")),
    base_model: str = typer.Option("yolo11s"),
    epochs: int = typer.Option(40),
    imgsz: int = typer.Option(640),
    batch: int = typer.Option(8),
    dry_run: bool = typer.Option(True),
    report: Path = typer.Option(Path("artifacts/reports/phase3-mixed-yolo11-training.json")),
) -> None:
    payload = train_mixed_yolo11(
        matrix_path=matrix,
        output_dir=output_dir,
        base_model=base_model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        dry_run=dry_run,
    )
    write_json(report, payload)
    typer.echo(f"Wrote Phase 3 YOLO11 training report to {report}")
