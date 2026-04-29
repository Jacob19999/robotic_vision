from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.synthetic.planner import write_synthetic_generation_plan


def plan_synthetic_generation_command(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    config: Path = typer.Option(Path("config/phase2.yaml"), exists=True, file_okay=True, dir_okay=False),
    output: Path = typer.Option(Path("artifacts/synthetic/phase2/plan.json")),
    phase1_summary: Optional[Path] = typer.Option(None, file_okay=True, dir_okay=False),
) -> None:
    payload = write_synthetic_generation_plan(
        manifest_path=manifest,
        config_path=config,
        output_path=output,
        phase1_summary_path=phase1_summary,
    )
    typer.echo(
        f"Wrote synthetic generation plan for {payload['total_frame_budget']} frames "
        f"across {len(payload['scenes'])} scene(s) to {output}"
    )
