from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.data.manifests.models import BenchmarkManifest, FailureExample
from src.evaluation.reporting.validator import validate_run_report
from src.utils.hardware import get_hardware_profile
from src.utils.paths import write_json


def build_run_report(
    *,
    run_id: str,
    model_family: str,
    run_mode: str,
    manifest: BenchmarkManifest,
    status: str,
    metrics: dict,
    failures: list[FailureExample],
    notes: str | None = None,
) -> dict:
    payload = {
        "run_id": run_id,
        "model_family": model_family,
        "run_mode": run_mode,
        "manifest_id": manifest.manifest_id,
        "status": status,
        "hardware_profile": get_hardware_profile(),
        "metrics": metrics,
        "failure_examples": [item.model_dump(mode="json") for item in failures],
        "notes": notes or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    validate_run_report(payload)
    return payload


def save_run_report(report: dict, output_path: str | Path) -> Path:
    return write_json(output_path, report)

