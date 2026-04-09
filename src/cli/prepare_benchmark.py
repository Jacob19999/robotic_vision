from __future__ import annotations

from pathlib import Path

import typer

from src.config.phase1_settings import load_phase1_config
from src.data.ingestion.normalize import normalize_assets
from src.data.ingestion.source_registry import load_source_assets
from src.data.manifests.builder import build_benchmark_manifest
from src.data.manifests.validator import validate_benchmark_manifest
from src.data.ontology.registry import OntologyRegistry
from src.utils.paths import write_json


def prepare_benchmark(config_path: str | Path, output_path: str | Path) -> dict:
    config, base_dir = load_phase1_config(config_path)
    registry = OntologyRegistry.from_config(config.ontology)
    raw_assets = []
    for source in config.sources:
        raw_assets.extend(load_source_assets(source, base_dir))
    normalized_assets = normalize_assets(raw_assets, registry, fail_on_unmapped=config.fail_on_unmapped)
    manifest = build_benchmark_manifest(normalized_assets, config, registry)
    payload = manifest.model_dump(mode="json", exclude_none=True)
    validate_benchmark_manifest(payload)
    write_json(output_path, payload)
    return payload


def prepare_benchmark_command(
    config: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output: Path = typer.Option(...),
) -> None:
    payload = prepare_benchmark(config, output)
    typer.echo(f"Wrote benchmark manifest {payload['manifest_id']} to {output}")
