from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.phase1_settings import BaselineRuntimeConfig, load_phase1_baseline_settings
from src.data.manifests.models import BenchmarkManifest
from src.models.common.predictions import AssetPrediction, ExecutionConfig, PredictedAnnotation, RunnerResult

_MODEL_REPOS = {
    "florence2-base": "microsoft/Florence-2-base",
    "florence2-base-ft": "microsoft/Florence-2-base-ft",
}


class Florence2Runner:
    def __init__(
        self,
        baseline_settings: BaselineRuntimeConfig | None = None,
        manifest_base_dir: Path | None = None,
    ) -> None:
        self.settings = baseline_settings or load_phase1_baseline_settings().florence2
        self.manifest_base_dir = manifest_base_dir
        # Precision-oriented defaults for open-vocabulary Florence outputs.
        self._max_detections_per_class = 2
        self._min_area_ratio = 0.002

    @staticmethod
    def _bbox_area_ratio(bbox: list[float], width: int, height: int) -> float:
        image_area = max(float(width * height), 1.0)
        box_width = max(float(bbox[2]) - float(bbox[0]), 0.0)
        box_height = max(float(bbox[3]) - float(bbox[1]), 0.0)
        return (box_width * box_height) / image_area

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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

        repo_id = _MODEL_REPOS.get(self.settings.model_variant, self.settings.model_variant)
        config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        for candidate in (config, getattr(config, "text_config", None), getattr(config, "language_config", None)):
            if candidate is not None and not hasattr(candidate, "forced_bos_token_id"):
                setattr(candidate, "forced_bos_token_id", None)
        processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True, config=config)
        torch_dtype = torch.float16 if self.settings.precision_mode.lower() == "fp16" else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            config=config,
        ).to(device)
        model.eval()
        return processor, model, torch

    @staticmethod
    def _text_prompt(manifest: BenchmarkManifest) -> str:
        labels: list[str] = []
        for item in manifest.classes:
            if item.status != "active":
                continue
            labels.append(item.canonical_name.strip())
        return ", ".join(labels)

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

    @staticmethod
    def _ensure_generation_compat(model: Any) -> None:
        """Patch known missing generation attrs on some Florence config objects."""
        config_candidates = [
            getattr(model, "config", None),
            getattr(getattr(model, "config", None), "text_config", None),
            getattr(model, "generation_config", None),
        ]
        for config in config_candidates:
            if config is None:
                continue
            if not hasattr(config, "forced_bos_token_id"):
                setattr(config, "forced_bos_token_id", None)

    def run(self, manifest: BenchmarkManifest) -> RunnerResult:
        execution_config = self._execution_config()
        try:
            from PIL import Image
        except ImportError as error:
            return RunnerResult(
                status="blocked",
                execution_config=execution_config,
                notes=f"Florence-2 execution is blocked because Pillow is unavailable: {error}",
            )

        try:
            processor, model, torch = self._load_runtime()
            self._ensure_generation_compat(model)
        except ImportError as error:
            return RunnerResult(
                status="blocked",
                execution_config=execution_config,
                notes=f"Florence-2 execution is blocked because Transformers/PyTorch is unavailable: {error}",
            )
        except Exception as error:
            return RunnerResult(
                status="failed",
                execution_config=execution_config,
                notes=f"Florence-2 model load failed: {error}",
            )

        task_prompt = "<OD>"
        label_lookup = self._label_lookup(manifest)
        predictions: list[AssetPrediction] = []

        for asset in manifest.assets:
            image_path = Path(asset.relative_path)
            if not image_path.exists() and self.manifest_base_dir is not None:
                image_path = (self.manifest_base_dir / asset.relative_path).resolve()
            if not image_path.exists():
                predictions.append(AssetPrediction(asset_id=asset.asset_id, predictions=[]))
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                # Florence-2 OD expects the task token alone; extra text breaks post-processing.
                model_inputs = processor(text=task_prompt, images=image, return_tensors="pt")
                device = next(model.parameters()).device
                model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
                model_dtype = next(model.parameters()).dtype
                if "pixel_values" in model_inputs:
                    model_inputs["pixel_values"] = model_inputs["pixel_values"].to(model_dtype)
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=model_inputs["input_ids"],
                        pixel_values=model_inputs.get("pixel_values"),
                        max_new_tokens=256,
                        num_beams=2,
                    )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=(image.width, image.height),
                )
                od = parsed.get(task_prompt, {}) if isinstance(parsed, dict) else {}
                boxes = od.get("bboxes", [])
                labels = od.get("labels", [])
                scores = od.get("scores", [])

                class_buckets: dict[str, list[PredictedAnnotation]] = {}
                for index, bbox in enumerate(boxes):
                    label = str(labels[index]).strip().lower() if index < len(labels) else ""
                    if ">" in label:
                        label = label.split(">")[-1].strip()
                    class_id = label_lookup.get(label)
                    if class_id is None:
                        continue
                    area_ratio = self._bbox_area_ratio([float(value) for value in bbox], image.width, image.height)
                    if area_ratio < self._min_area_ratio:
                        continue
                    score = float(scores[index]) if index < len(scores) else 1.0
                    prediction = PredictedAnnotation(
                        class_id=class_id,
                        bbox_xyxy=[float(value) for value in bbox],
                        score=score,
                    )
                    class_buckets.setdefault(class_id, []).append(prediction)

                asset_predictions: list[PredictedAnnotation] = []
                for class_id, bucket in class_buckets.items():
                    # Keep top largest boxes per class to suppress dense false positives.
                    ranked = sorted(
                        bucket,
                        key=lambda item: self._bbox_area_ratio(item.bbox_xyxy, image.width, image.height),
                        reverse=True,
                    )
                    asset_predictions.extend(ranked[: self._max_detections_per_class])
                predictions.append(AssetPrediction(asset_id=asset.asset_id, predictions=asset_predictions))
            except Exception:
                predictions.append(AssetPrediction(asset_id=asset.asset_id, predictions=[]))

        return RunnerResult(
            status="completed",
            execution_config=execution_config,
            predictions=predictions,
            notes=f"Florence-2 executed with {_MODEL_REPOS.get(self.settings.model_variant, self.settings.model_variant)}.",
        )
