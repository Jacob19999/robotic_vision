from __future__ import annotations

import typer

from src.cli.export_mixed_yolo_experiments import export_mixed_yolo_experiments_command
from src.cli.export_yolo_view import export_yolo_view_command
from src.cli.export_yolo_book_oversampled_view import export_yolo_book_oversampled_view_command
from src.cli.plan_synthetic_generation import plan_synthetic_generation_command
from src.cli.prepare_benchmark import prepare_benchmark_command
from src.cli.run_baseline import run_baseline_command
from src.cli.summarize_phase1 import summarize_phase1_command
from src.cli.train_florence2 import train_florence2_command
from src.cli.train_mixed_yolo11 import train_mixed_yolo11_command
from src.cli.train_yolo11_book_focus import train_yolo11_book_focus_command
from src.cli.write_isaac_synthetic_script import write_isaac_synthetic_script_command


app = typer.Typer(help="Robotic vision staged detection pipeline CLI.")

app.command("prepare-benchmark")(prepare_benchmark_command)
app.command("export-yolo-view")(export_yolo_view_command)
app.command("export-yolo-book-oversampled-view")(export_yolo_book_oversampled_view_command)
app.command("run-baseline")(run_baseline_command)
app.command("summarize-phase1")(summarize_phase1_command)
app.command("train-florence2")(train_florence2_command)
app.command("train-yolo11-book-focus")(train_yolo11_book_focus_command)
app.command("plan-synthetic-generation")(plan_synthetic_generation_command)
app.command("write-isaac-synthetic-script")(write_isaac_synthetic_script_command)
app.command("export-mixed-yolo-experiments")(export_mixed_yolo_experiments_command)
app.command("train-mixed-yolo11")(train_mixed_yolo11_command)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
