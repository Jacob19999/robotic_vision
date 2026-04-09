from __future__ import annotations

import shutil
import subprocess
from typing import Any


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
        "gpu_vram_mb": int(float(memory)),
    }


def get_hardware_profile() -> dict[str, Any]:
    profile = _nvidia_smi_profile()
    if not profile:
        profile = {
            "gpu_name": "unknown",
            "gpu_vram_mb": 0,
        }
    try:
        import psutil

        profile["system_ram_mb"] = round(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        profile["system_ram_mb"] = 0
    return profile

