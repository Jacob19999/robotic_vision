from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def _git_feature_name() -> str | None:
    if not (PROJECT_ROOT / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    if branch and (PROJECT_ROOT / "specs" / branch).exists():
        return branch
    return None


def get_feature_dir(feature_name: str | None = None) -> Path:
    """Resolve the active feature directory under specs/.

    Resolution order:
    1. explicit ``feature_name``
    2. ``SPECIFY_FEATURE`` environment variable
    3. current git branch when it matches a folder under ``specs/``
    4. ``001-phase1-baseline`` when present
    5. the lexicographically latest feature directory
    """
    specs_dir = PROJECT_ROOT / "specs"
    selected = feature_name or os.environ.get("SPECIFY_FEATURE") or _git_feature_name()
    if selected:
        candidate = specs_dir / selected
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Feature directory not found: {candidate}")

    default_baseline = specs_dir / "001-phase1-baseline"
    if default_baseline.exists():
        return default_baseline

    candidates = sorted([item for item in specs_dir.iterdir() if item.is_dir()]) if specs_dir.exists() else []
    if not candidates:
        raise FileNotFoundError("No feature directory found under specs/.")
    return candidates[-1]


def get_contracts_dir(feature_name: str | None = None) -> Path:
    return get_feature_dir(feature_name) / "contracts"


def get_contract_path(filename: str, feature_name: str | None = None) -> Path:
    preferred = get_contracts_dir(feature_name) / filename
    if preferred.exists():
        return preferred

    specs_dir = PROJECT_ROOT / "specs"
    if specs_dir.exists():
        for feature_dir in sorted([item for item in specs_dir.iterdir() if item.is_dir()], reverse=True):
            candidate = feature_dir / "contracts" / filename
            if candidate.exists():
                return candidate

    return preferred


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
