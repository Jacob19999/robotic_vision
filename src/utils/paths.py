from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_feature_dir(feature_name: str | None = None) -> Path:
    specs_dir = PROJECT_ROOT / "specs"
    selected = feature_name or os.environ.get("SPECIFY_FEATURE")
    if selected:
        candidate = specs_dir / selected
        if candidate.exists():
            return candidate

    candidates = sorted([item for item in specs_dir.iterdir() if item.is_dir()]) if specs_dir.exists() else []
    if not candidates:
        raise FileNotFoundError("No feature directory found under specs/.")
    return candidates[-1]


def get_contracts_dir(feature_name: str | None = None) -> Path:
    return get_feature_dir(feature_name) / "contracts"


def get_contract_path(filename: str, feature_name: str | None = None) -> Path:
    return get_contracts_dir(feature_name) / filename


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = ensure_parent(path)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return resolved


def artifact_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath("artifacts", *parts)

