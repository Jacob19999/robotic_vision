from __future__ import annotations

import shutil
import subprocess
from typing import Any


def _positive_mb(value: int | float | None, default: int = 1) -> int:
    if value is None:
        return default
    return max(int(round(float(value))), default)


def _nvidia_smi_profile() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    name, memory = [part.strip() for part in result.stdout.strip().split(",", maxsplit=1)]
    return {
        "gpu_name": name,
        "gpu_vram_mb": _positive_mb(memory),
    }


def get_hardware_profile() -> dict[str, Any]:
    profile = _nvidia_smi_profile()
    if not profile:
        profile = {
            "gpu_name": "unknown",
            "gpu_vram_mb": 1,
        }
    try:
        import psutil

        profile["system_ram_mb"] = _positive_mb(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        profile["system_ram_mb"] = 1
    return profile
