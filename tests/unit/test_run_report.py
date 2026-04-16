from __future__ import annotations

from src.data.manifests.models import BenchmarkManifest
from src.evaluation.reporting.run_report import build_run_report
from src.models.common.predictions import ExecutionConfig
from src.utils.hardware import get_hardware_profile
from src.utils.paths import read_json


def test_build_run_report_captures_execution_config(repo_root) -> None:  # noqa: ANN001
    manifest = BenchmarkManifest.model_validate(
        read_json(repo_root / "tests" / "fixtures" / "phase1_reports" / "sample_run_input.json")
    )
    report = build_run_report(
        run_id="run-yolo",
        model_family="yolo11",
        run_mode="trainable",
        manifest=manifest,
        status="blocked",
        execution_config=ExecutionConfig(
            model_variant="yolo11n",
            resolution=640,
            precision_mode="fp16",
            batch_size=1,
            seed=42,
            dataset_view_id="view-001",
            checkpoint_reference="yolo11n.pt",
        ),
        metrics={"latency_ms_per_image": 0.0, "peak_vram_mb": 0.0},
        failures=[],
        notes="hardware constrained",
        hardware_profile={"gpu_name": "unknown", "gpu_vram_mb": 1, "system_ram_mb": 1},
    )

    assert report["execution_config"]["dataset_view_id"] == "view-001"
    assert report["hardware_profile"]["gpu_vram_mb"] == 1


def test_get_hardware_profile_remains_schema_compliant() -> None:
    profile = get_hardware_profile()

    assert profile["gpu_name"]
    assert profile["gpu_vram_mb"] >= 1
    assert profile["system_ram_mb"] >= 1
