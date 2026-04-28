from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config.phase1_settings import BaselineRuntimeConfig, load_phase1_baseline_settings
from src.data.manifests.models import BenchmarkManifest, DatasetAsset
from src.data.views.models import DetectorDatasetView
from src.data.views.yolo import exported_asset_name
from src.models.common.predictions import AssetPrediction, ExecutionConfig, PredictedAnnotation, RunnerResult


def _default_model_loader(model_variant: str) -> Any:
    from ultralytics import YOLO

    candidate = Path(model_variant)
    model_source = str(candidate) if candidate.exists() else f"{model_variant}.pt"
    return YOLO(model_source)


def _is_hardware_limit(error: Exception) -> bool:
    message = str(error).lower()
    hardware_markers = (
        "out of memory",
        "cuda",
        "vram",
        "insufficient memory",
        "not enough memory",
        "mps backend out of memory",
    )
    return any(marker in message for marker in hardware_markers)


def _values_to_list(values: Any) -> list[Any]:
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    return [values]


class YOLO11Runner:
    def __init__(
        self,
        dataset_view: DetectorDatasetView,
        *,
        model_loader: Callable[[str], Any] | None = None,
        baseline_settings: BaselineRuntimeConfig | None = None,
    ) -> None:
        self.dataset_view = dataset_view
        self.model_loader = model_loader or _default_model_loader
        self.settings = baseline_settings or load_phase1_baseline_settings().yolo11
        self._ontology_label_map = self._build_ontology_label_map()
        self._trained_best_checkpoint = (
            Path("runs/detect/artifacts/training/yolo11-fulltrain-gpu/weights/best.pt").resolve()
        )

    def _build_ontology_label_map(self) -> dict[str, str]:
        """Map canonical/alias labels to ontology class IDs."""
        label_map: dict[str, str] = {}
        for class_id in self.dataset_view.class_order:
            label_map[class_id.lower()] = class_id
        # Common COCO-to-ontology normalization used in the baseline config.
        if "mug" in self.dataset_view.class_order:
            label_map.setdefault("cup", "mug")
        return label_map

    def _resolve_ontology_class(self, result: Any, class_index: int) -> str | None:
        names = getattr(result, "names", None)
        if names is None:
            return None
        raw_label = None
        if isinstance(names, dict):
            raw_label = names.get(class_index)
        elif isinstance(names, list) and 0 <= class_index < len(names):
            raw_label = names[class_index]
        if not raw_label:
            return None

        normalized = str(raw_label).strip().lower()
        return self._ontology_label_map.get(normalized)

    def _execution_config(self, model_variant: str, checkpoint_reference: str) -> ExecutionConfig:
        return ExecutionConfig(
            model_variant=model_variant,
            resolution=self.settings.resolution,
            precision_mode=self.settings.precision_mode,
            batch_size=self.settings.batch_size,
            seed=self.settings.seed,
            dataset_view_id=self.dataset_view.view_id,
            checkpoint_reference=checkpoint_reference,
        )

    def _resolve_primary_checkpoint_reference(self, model_variant: str) -> str:
        if self._trained_best_checkpoint.exists():
            return str(self._trained_best_checkpoint)
        return f"{model_variant}.pt"

    def _resolve_image_path(self, asset: DatasetAsset) -> Path:
        split_export = self.dataset_view.split_exports[asset.split_name or "train"]
        candidate = Path(split_export.images_dir) / exported_asset_name(asset)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Exported image not found for asset {asset.asset_id}: {candidate}")

    def _predict_for_asset(self, model: Any, asset: DatasetAsset) -> AssetPrediction:
        image_path = self._resolve_image_path(asset)
        results = model.predict(
            source=str(image_path),
            imgsz=self.settings.resolution,
            verbose=False,
            batch=self.settings.batch_size,
            half=self.settings.precision_mode.lower() == "fp16",
        )
        result = results[0] if isinstance(results, list) else results
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return AssetPrediction(asset_id=asset.asset_id, predictions=[])

        xyxy_values = _values_to_list(getattr(boxes, "xyxy", []))
        class_values = _values_to_list(getattr(boxes, "cls", []))
        score_values = _values_to_list(getattr(boxes, "conf", []))

        predictions: list[PredictedAnnotation] = []
        for index, bbox in enumerate(xyxy_values):
            class_index = int(class_values[index]) if index < len(class_values) else 0
            class_id = self._resolve_ontology_class(result, class_index)
            if class_id is None:
                continue
            score = float(score_values[index]) if index < len(score_values) else 1.0
            predictions.append(
                PredictedAnnotation(
                    class_id=class_id,
                    bbox_xyxy=[float(value) for value in bbox],
                    score=score,
                )
            )
        return AssetPrediction(asset_id=asset.asset_id, predictions=predictions)

    def _run_variant(
        self,
        manifest: BenchmarkManifest,
        model_variant: str,
        checkpoint_reference: str,
    ) -> RunnerResult:
        loader_input = checkpoint_reference if self.model_loader is _default_model_loader else model_variant
        model = self.model_loader(loader_input)
        predictions = [self._predict_for_asset(model, asset) for asset in manifest.assets]
        return RunnerResult(
            status="completed",
            execution_config=self._execution_config(model_variant, checkpoint_reference),
            predictions=predictions,
            notes=f"YOLO11 executed with {checkpoint_reference}.",
        )

    def run(self, manifest: BenchmarkManifest) -> RunnerResult:
        primary_variant = self.settings.model_variant
        fallback_variant = self.settings.fallback_variant
        primary_checkpoint = self._resolve_primary_checkpoint_reference(primary_variant)
        fallback_checkpoint = f"{fallback_variant}.pt" if fallback_variant else ""
        fallback_reason: str | None = None

        try:
            return self._run_variant(manifest, primary_variant, primary_checkpoint)
        except ImportError as error:
            return RunnerResult(
                status="blocked",
                execution_config=self._execution_config(primary_variant, primary_checkpoint),
                notes=f"YOLO11 execution is blocked because Ultralytics is unavailable: {error}",
            )
        except FileNotFoundError as error:
            return RunnerResult(
                status="blocked",
                execution_config=self._execution_config(primary_variant, primary_checkpoint),
                notes=str(error),
            )
        except Exception as error:
            if not fallback_variant or fallback_variant == primary_variant or not _is_hardware_limit(error):
                return RunnerResult(
                    status="failed",
                    execution_config=self._execution_config(primary_variant, primary_checkpoint),
                    notes=f"YOLO11 execution failed for {primary_variant}: {error}",
                )
            fallback_reason = str(error)

        try:
            result = self._run_variant(manifest, fallback_variant, fallback_checkpoint)
            result.notes = (
                f"YOLO11 fell back from {primary_variant} to {fallback_variant} after a hardware limit: "
                f"{fallback_reason}"
            )
            return result
        except Exception as error:
            status = "blocked" if _is_hardware_limit(error) else "failed"
            return RunnerResult(
                status=status,
                execution_config=self._execution_config(fallback_variant, fallback_checkpoint),
                notes=(
                    f"YOLO11 could not complete with {primary_variant} ({fallback_reason}) "
                    f"or {fallback_variant} ({error})."
                ),
            )
