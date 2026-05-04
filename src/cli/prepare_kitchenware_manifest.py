from __future__ import annotations

from pathlib import Path

import typer

from src.data.manifests.kitchenware import build_kitchenware_manifest


def prepare_kitchenware_manifest_command(
    dataset_root: Path = typer.Option(
        ...,
        exists=True,
        file_okay=False,
        help="Folder with train.csv and images/ (Kaggle kitchenware-classification unzip layout).",
    ),
    output: Path = typer.Option(Path("artifacts/manifests/kitchenware-benchmark.json")),
    manifest_id: str = typer.Option("kitchenware-kaggle"),
    seed: int = typer.Option(42),
    train_ratio: float = typer.Option(0.75),
    val_ratio: float = typer.Option(0.125),
    max_rows: int | None = typer.Option(
        None,
        help="Optional cap on train.csv rows (head) for quick experiments.",
    ),
) -> None:
    output_path = output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_kitchenware_manifest(
        dataset_root,
        manifest_id=manifest_id,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        max_rows=max_rows,
        output_path=output_path,
    )
    typer.echo(
        f"Wrote manifest {manifest.manifest_id} with {len(manifest.assets)} assets to {output_path}"
    )
