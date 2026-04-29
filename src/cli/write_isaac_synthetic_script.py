from __future__ import annotations

from pathlib import Path

import typer

from src.synthetic.isaac_script import write_isaac_replicator_script
from src.synthetic.models import SyntheticGenerationPlan
from src.utils.paths import read_json


def write_isaac_synthetic_script_command(
    plan: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output: Path = typer.Option(Path("artifacts/synthetic/phase2/generate_phase2_replicator.py")),
) -> None:
    payload = read_json(plan)
    write_isaac_replicator_script(SyntheticGenerationPlan.model_validate(payload), output)
    typer.echo(f"Wrote Isaac Sim Replicator script to {output}")

