from __future__ import annotations

from src.data.manifests.validator import validate_benchmark_manifest
from src.evaluation.reporting.validator import validate_run_report


def test_benchmark_manifest_schema_accepts_valid_payload() -> None:
    payload = {
        "manifest_id": "phase1-sample",
        "ontology_version": "v1",
        "source_ids": ["coco2017"],
        "split_versions": {
            "train": "train-v1",
            "val": "val-v1",
            "test_real_heldout": "test-v1",
        },
        "asset_counts": {
            "train": 1,
            "val": 0,
            "test_real_heldout": 0,
            "by_class": {"mug": 1},
        },
        "created_at": "2026-04-08T00:00:00+00:00",
    }
    assert validate_benchmark_manifest(payload) == payload


def test_run_report_schema_accepts_valid_payload() -> None:
    payload = {
        "run_id": "run-001",
        "model_family": "grounding_dino",
        "run_mode": "zero_shot",
        "manifest_id": "phase1-sample",
        "status": "completed",
        "hardware_profile": {
            "gpu_name": "Example GPU",
            "gpu_vram_mb": 1024,
            "system_ram_mb": 8192,
        },
        "metrics": {
            "latency_ms_per_image": 12.5,
            "peak_vram_mb": 900,
            "mAP": 0.5,
        },
        "created_at": "2026-04-08T00:00:00+00:00",
    }
    assert validate_run_report(payload) == payload

