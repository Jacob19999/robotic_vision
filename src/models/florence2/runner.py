from __future__ import annotations

from collections.abc import Callable

from src.data.manifests.models import BenchmarkManifest
from src.models.common.predictions import RunnerResult


class Florence2Runner:
    def __init__(self, predictor: Callable[[BenchmarkManifest], RunnerResult] | None = None) -> None:
        self.predictor = predictor

    def run(self, manifest: BenchmarkManifest) -> RunnerResult:
        if self.predictor is not None:
            return self.predictor(manifest)
        return RunnerResult(
            status="blocked",
            notes="Florence-2 execution requires explicit model wiring or a test predictor.",
        )

