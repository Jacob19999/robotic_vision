from __future__ import annotations

import time
from pathlib import Path

import typer

from src.data.manifests.models import BenchmarkManifest
from src.evaluation.metrics.service import compute_detection_metrics
from src.evaluation.reporting.artifacts import export_failure_examples
from src.evaluation.reporting.run_report import build_run_report, save_run_report
from src.models.florence2.runner import Florence2Runner
from src.models.grounding_dino.runner import GroundingDINORunner
from src.utils.paths import artifact_path, read_json


RUNNER_FACTORIES = {
    "grounding_dino": GroundingDINORunner,
    "florence2": Florence2Runner,
}


def run_baseline(model: str, manifest_path: str | Path, report_path: str | Path) -> dict:
    manifest = BenchmarkManifest.model_validate(read_json(manifest_path))
    if model not in RUNNER_FACTORIES:
        raise ValueError(f"Unsupported model family: {model}")

    runner = RUNNER_FACTORIES[model]()
    started = time.perf_counter()
    result = runner.run(manifest)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 4)

    if result.status != "completed":
        report = build_run_report(
            run_id=f"{model}-{int(time.time())}",
            model_family=model,
            run_mode="zero_shot" if model == "grounding_dino" else "trainable",
            manifest=manifest,
            status=result.status,
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
        run_mode="zero_shot" if model == "grounding_dino" else "trainable",
        manifest=manifest,
        status="completed",
        metrics=metrics,
        failures=failures,
        notes=result.notes,
    )
    save_run_report(report, report_path)
    return report


def run_baseline_command(
    model: str = typer.Option(..., help="grounding_dino or florence2"),
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    report: Path = typer.Option(...),
) -> None:
    payload = run_baseline(model, manifest, report)
    typer.echo(f"Wrote baseline report {payload['run_id']} to {report}")

