from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from src.utils.paths import get_contract_path, read_json


def validate_detector_dataset_view(
    payload: dict[str, Any],
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    schema = read_json(schema_path or get_contract_path("detector-dataset-view.schema.json"))
    Draft202012Validator(schema).validate(payload)
    return payload
