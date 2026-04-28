from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer


def _to_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def train_yolo11_book_focus(
    *,
    dataset_yaml: str | Path,
    base_checkpoint: str | Path,
    output_dir: str | Path,
    output_name: str = "yolo11-book-focus",
    epochs: int = 40,
    imgsz: int = 640,
    batch: int = 8,
    patience: int = 20,
) -> dict:
    from ultralytics import YOLO

    dataset_yaml_path = Path(dataset_yaml).resolve()
    checkpoint_path = Path(base_checkpoint).resolve()
    output_dir_path = Path(output_dir).resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(checkpoint_path))
    model.train(
        data=str(dataset_yaml_path),
        project=str(output_dir_path),
        name=output_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        cls=1.5,
        mosaic=0.7,
        mixup=0.2,
        copy_paste=0.15,
        fliplr=0.5,
        degrees=5.0,
        scale=0.25,
        translate=0.1,
        save=True,
    )

    best_checkpoint = output_dir_path / output_name / "weights" / "best.pt"
    tuned_model = YOLO(str(best_checkpoint))
    # Ultralytics resolves held-out images from dataset YAML through the "test" split key.
    metrics = tuned_model.val(data=str(dataset_yaml_path), split="test")

    names = getattr(metrics, "names", {}) or {}
    per_class_ap50_95: dict[str, float] = {}
    for index, name in names.items():
        per_class_ap50_95[str(name)] = _to_float(metrics.box.maps[int(index)])

    report = {
        "report_type": "yolo11_book_focus_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(best_checkpoint),
        "base_checkpoint": str(checkpoint_path),
        "dataset_yaml": str(dataset_yaml_path),
        "split": "test",
        "training": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "patience": patience,
            "optimizer": "AdamW",
            "cls_gain": 1.5,
            "augmentations": {
                "mosaic": 0.7,
                "mixup": 0.2,
                "copy_paste": 0.15,
                "fliplr": 0.5,
                "degrees": 5.0,
                "scale": 0.25,
                "translate": 0.1,
            },
        },
        "summary_metrics": {
            "precision": _to_float(metrics.box.mp),
            "recall": _to_float(metrics.box.mr),
            "mAP50": _to_float(metrics.box.map50),
            "mAP50_95": _to_float(metrics.box.map),
        },
        "per_class_mAP50_95": per_class_ap50_95,
    }
    return report


def _load_eval_report(path: str | Path) -> dict:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_eval_comparison(current_report: dict, baseline_report: dict) -> dict:
    current_summary = current_report.get("summary_metrics", {})
    baseline_summary = baseline_report.get("summary_metrics", {})
    current_class = current_report.get("per_class_mAP50_95", {})
    baseline_class = baseline_report.get("per_class_mAP50_95", {})

    def delta(metric: str) -> float:
        return _to_float(current_summary.get(metric, 0.0)) - _to_float(baseline_summary.get(metric, 0.0))

    return {
        "report_type": "yolo11_eval_comparison",
        "current_checkpoint": current_report.get("checkpoint"),
        "baseline_checkpoint": baseline_report.get("checkpoint"),
        "delta": {
            "precision": delta("precision"),
            "recall": delta("recall"),
            "mAP50": delta("mAP50"),
            "mAP50_95": delta("mAP50_95"),
            "book_mAP50_95": _to_float(current_class.get("book", 0.0)) - _to_float(baseline_class.get("book", 0.0)),
            "mug_mAP50_95": _to_float(current_class.get("mug", 0.0)) - _to_float(baseline_class.get("mug", 0.0)),
        },
    }


def train_yolo11_book_focus_command(
    dataset_yaml: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    base_checkpoint: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(Path("runs/detect/artifacts/training")),
    output_name: str = typer.Option("yolo11-book-focus"),
    epochs: int = typer.Option(40),
    imgsz: int = typer.Option(640),
    batch: int = typer.Option(8),
    patience: int = typer.Option(20),
    report: Path = typer.Option(Path("artifacts/reports/yolo11-book-focus-eval.json")),
    baseline_report: Path = typer.Option(Path("artifacts/reports/yolo11-best-eval.json")),
    comparison_report: Path = typer.Option(Path("artifacts/reports/yolo11-book-focus-vs-best.json")),
) -> None:
    from src.utils.paths import write_json

    payload = train_yolo11_book_focus(
        dataset_yaml=dataset_yaml,
        base_checkpoint=base_checkpoint,
        output_dir=output_dir,
        output_name=output_name,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
    )
    write_json(report, payload)
    if baseline_report.exists():
        baseline_payload = _load_eval_report(baseline_report)
        comparison_payload = build_eval_comparison(payload, baseline_payload)
        write_json(comparison_report, comparison_payload)
    typer.echo(f"Wrote book-focused YOLO11 report to {report}")
