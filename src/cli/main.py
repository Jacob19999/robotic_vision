from __future__ import annotations

import typer

from src.cli.export_yolo_view import export_yolo_view_command
from src.cli.export_yolo_book_oversampled_view import export_yolo_book_oversampled_view_command
from src.cli.prepare_benchmark import prepare_benchmark_command
from src.cli.run_baseline import run_baseline_command
from src.cli.summarize_phase1 import summarize_phase1_command
from src.cli.train_yolo11_book_focus import train_yolo11_book_focus_command


app = typer.Typer(help="Phase 1 baseline pipeline CLI.")

app.command("prepare-benchmark")(prepare_benchmark_command)
app.command("export-yolo-view")(export_yolo_view_command)
app.command("export-yolo-book-oversampled-view")(export_yolo_book_oversampled_view_command)
app.command("run-baseline")(run_baseline_command)
app.command("summarize-phase1")(summarize_phase1_command)
app.command("train-yolo11-book-focus")(train_yolo11_book_focus_command)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
