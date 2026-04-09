from __future__ import annotations

import json
from pathlib import Path

from src.data.manifests.models import FailureExample
from src.utils.paths import ensure_directory


def export_failure_examples(failures: list[FailureExample], output_dir: str | Path) -> list[FailureExample]:
    destination = ensure_directory(output_dir)
    for failure in failures:
        artifact_path = destination / f"{failure.failure_id}.json"
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(failure.model_dump(mode="json"), handle, indent=2)
        failure.artifact_path = str(artifact_path)
    return failures

