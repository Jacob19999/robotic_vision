from __future__ import annotations

import os
from pathlib import Path

import pytest


os.environ["SPECIFY_FEATURE"] = "002-yolo-baseline-validation"


@pytest.fixture()
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def temp_artifact_dir(tmp_path: Path) -> Path:
    path = tmp_path / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path
