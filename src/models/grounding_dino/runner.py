from __future__ import annotations

from collections.abc import Callable

from src.data.manifests.models import BenchmarkManifest
from src.models.common.predictions import ExecutionConfig, RunnerResult


class GroundingDINORunner:
    def __init__(self, predictor: Callable[[BenchmarkManifest], RunnerResult] | None = None) -> None:
        self.predictor = predictor

    def run(self, manifest: BenchmarkManifest) -> RunnerResult:
        if self.predictor is not None:
            return self.predictor(manifest)
        return RunnerResult(
            status="blocked",
            execution_config=ExecutionConfig(
                model_variant="grounding-dino-base",
                resolution=640,
                precision_mode="fp16",
                batch_size=1,
            ),
            notes="Grounding DINO execution requires explicit model wiring or a test predictor.",
        )
