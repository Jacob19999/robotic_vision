from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.phase1_settings import BaselineRuntimeConfig, load_phase1_baseline_settings
from src.data.manifests.models import BenchmarkManifest
from src.models.common.predictions import AssetPrediction, ExecutionConfig, PredictedAnnotation, RunnerResult

_MODEL_REPOS = {
    "grounding-dino-base": "IDEA-Research/grounding-dino-base",
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
}


class GroundingDINORunner:
    def __init__(self, baseline_settings: BaselineRuntimeConfig | None = None) -> None:
        self.settings = baseline_settings or load_phase1_baseline_settings().grounding_dino

    def _execution_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            model_variant=self.settings.model_variant,
            resolution=self.settings.resolution,
            precision_mode=self.settings.precision_mode,
            batch_size=self.settings.batch_size,
            seed=self.settings.seed,
            checkpoint_reference=_MODEL_REPOS.get(self.settings.model_variant, self.settings.model_variant),
        )

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        repo_id = _MODEL_REPOS.get(self.settings.model_variant, self.settings.model_variant)
        processor = AutoProcessor.from_pretrained(repo_id)
        torch_dtype = torch.float16 if self.settings.precision_mode.lower() == "fp16" else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForZeroShotObjectDetection.from_pretrained(repo_id, torch_dtype=torch_dtype).to(device).eval()
        return processor, model, torch

    @staticmethod
    def _build_text_prompt(manifest: BenchmarkManifest) -> str:
        label_terms: list[str] = []
        for item in manifest.classes:
            if item.status != "active":
                continue
            label_terms.append(item.canonical_name.strip().lower())
            label_terms.extend(alias.strip().lower() for alias in item.aliases if alias.strip())
        # Grounding DINO works best with period-delimited categories.
        deduped = list(dict.fromkeys(label_terms))
        return ". ".join(deduped) + "."

    @staticmethod
    def _label_lookup(manifest: BenchmarkManifest) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for item in manifest.classes:
            if item.status != "active":
                continue
            lookup[item.canonical_name.strip().lower()] = item.class_id
            for alias in item.aliases:
                cleaned = alias.strip().lower()
                if cleaned:
                    lookup[cleaned] = item.class_id
        return lookup

    def run(self, manifest: BenchmarkManifest) -> RunnerResult:
        execution_config = self._execution_config()
        try:
            from PIL import Image
        except ImportError as error:
            return RunnerResult(
                status="blocked",
                execution_config=execution_config,
                notes=f"Grounding DINO execution is blocked because Pillow is unavailable: {error}",
            )

        try:
            processor, model, torch = self._load_runtime()
        except ImportError as error:
            return RunnerResult(
                status="blocked",
                execution_config=execution_config,
                notes=f"Grounding DINO execution is blocked because Transformers/PyTorch is unavailable: {error}",
            )
        except Exception as error:
            return RunnerResult(
                status="failed",
                execution_config=execution_config,
                notes=f"Grounding DINO model load failed: {error}",
            )

        text_prompt = self._build_text_prompt(manifest)
        label_lookup = self._label_lookup(manifest)
        predictions: list[AssetPrediction] = []

        for asset in manifest.assets:
            image_path = Path(asset.relative_path)
            if not image_path.exists():
                predictions.append(AssetPrediction(asset_id=asset.asset_id, predictions=[]))
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, text=text_prompt, return_tensors="pt")
                device = next(model.parameters()).device
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                processed = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs["input_ids"],
                    box_threshold=0.25,
                    text_threshold=0.25,
                    target_sizes=[image.size[::-1]],
                )[0]
                asset_predictions: list[PredictedAnnotation] = []
                for score, label, bbox in zip(
                    processed.get("scores", []),
                    processed.get("labels", []),
                    processed.get("boxes", []),
                    strict=False,
                ):
                    normalized_label = str(label).strip().lower()
                    class_id = label_lookup.get(normalized_label)
                    if class_id is None:
                        continue
                    values = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
                    asset_predictions.append(
                        PredictedAnnotation(
                            class_id=class_id,
                            bbox_xyxy=[float(value) for value in values],
                            score=float(score),
                        )
                    )
                predictions.append(AssetPrediction(asset_id=asset.asset_id, predictions=asset_predictions))
            except Exception:
                predictions.append(AssetPrediction(asset_id=asset.asset_id, predictions=[]))

        return RunnerResult(
            status="completed",
            execution_config=execution_config,
            predictions=predictions,
            notes=f"Grounding DINO executed with {_MODEL_REPOS.get(self.settings.model_variant, self.settings.model_variant)}.",
        )
