from __future__ import annotations

from pathlib import Path

import yaml

from src.cli.prepare_benchmark import prepare_benchmark
from src.utils.paths import read_json


def test_prepare_benchmark_builds_manifest(repo_root: Path, tmp_path: Path) -> None:
    config_path = tmp_path / "phase1.yaml"
    config = {
        "manifest_id": "phase1-sample",
        "ontology_version": "v1",
        "ontology": [
            {"class_id": "mug", "canonical_name": "mug", "aliases": ["cup"]},
            {"class_id": "book", "canonical_name": "book", "aliases": []},
        ],
        "sources": [
            {
                "source_id": "coco2017",
                "name": "COCO 2017 sample",
                "source_type": "public_curated",
                "format": "coco",
                "path": str(repo_root / "tests" / "fixtures" / "phase1_benchmark" / "manifest_input.json"),
            }
        ],
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    output_path = tmp_path / "benchmark.json"
    payload = prepare_benchmark(config_path, output_path)
    saved = read_json(output_path)

    assert payload["manifest_id"] == "phase1-sample"
    assert saved["asset_counts"]["train"] + saved["asset_counts"]["val"] + saved["asset_counts"]["test_real_heldout"] == 2
    assert {item["class_id"] for item in saved["classes"]} == {"mug", "book"}
