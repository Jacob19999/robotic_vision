from __future__ import annotations

from pathlib import Path

from src.utils.paths import write_json


def export_failure_summary(grouped_failures: list[dict], output_path: str | Path) -> Path:
    payload = {"top_failures": grouped_failures}
    return write_json(output_path, payload)

