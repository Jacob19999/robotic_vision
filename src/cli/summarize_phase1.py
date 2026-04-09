from __future__ import annotations

from pathlib import Path

import typer

from src.evaluation.failure_analysis.export import export_failure_summary
from src.evaluation.reporting.phase1_summary import build_phase1_summary
from src.utils.paths import read_json, write_json


def summarize_phase1(reports: str | Path, output_path: str | Path) -> dict:
    reports_path = Path(reports)
    report_files = sorted(reports_path.glob("*.json")) if reports_path.is_dir() else [reports_path]
    payloads = [read_json(path) for path in report_files]
    summary = build_phase1_summary(payloads)
    write_json(output_path, summary)
    export_failure_summary(summary["top_failures"], Path(output_path).with_name("phase1-failures.json"))
    return summary


def summarize_phase1_command(
    reports: Path = typer.Option(..., exists=True),
    output: Path = typer.Option(...),
) -> None:
    payload = summarize_phase1(reports, output)
    typer.echo(f"Wrote Phase 1 summary {payload['summary_id']} to {output}")

