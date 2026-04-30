from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time

import typer

from src.config.phase1_settings import BaselineRuntimeConfig
from src.data.manifests.models import BenchmarkManifest, DatasetAsset
from src.evaluation.metrics.service import compute_detection_metrics
from src.models.florence2.runner import Florence2Runner
from src.utils.paths import read_json, write_json

_OD_PROMPT = "<OD>"


@dataclass
class _Sample:
    asset: DatasetAsset
    image_path: Path


def _resolve_image_path(asset: DatasetAsset, manifest_base_dir: Path) -> Path | None:
    candidate = Path(asset.relative_path)
    if candidate.exists():
        return candidate.resolve()
    fallback = (manifest_base_dir / asset.relative_path).resolve()
    if fallback.exists():
        return fallback
    return None


def _loc_token(value: float, max_value: int) -> str:
    if max_value <= 0:
        index = 0
    else:
        normalized = max(0.0, min(float(value) / float(max_value), 1.0))
        index = int(round(normalized * 999))
    return f"<loc_{index}>"


def _build_detection_target(asset: DatasetAsset, class_name_by_id: dict[str, str]) -> str:
    chunks: list[str] = []
    for annotation in asset.annotations:
        if annotation.is_ignored:
            continue
        class_name = class_name_by_id.get(annotation.class_id, annotation.class_id)
        x1, y1, x2, y2 = annotation.bbox_xyxy
        chunks.append(
            "".join(
                (
                    _loc_token(x1, asset.width),
                    _loc_token(y1, asset.height),
                    _loc_token(x2, asset.width),
                    _loc_token(y2, asset.height),
                    class_name,
                )
            )
        )
    return "".join(chunks)


def _build_samples(manifest: BenchmarkManifest, manifest_base_dir: Path, split_name: str) -> list[_Sample]:
    samples: list[_Sample] = []
    for asset in manifest.assets:
        if asset.split_name != split_name:
            continue
        image_path = _resolve_image_path(asset, manifest_base_dir)
        if image_path is None:
            continue
        samples.append(_Sample(asset=asset, image_path=image_path))
    return samples


def _evaluate_checkpoint(
    *,
    manifest: BenchmarkManifest,
    manifest_base_dir: Path,
    checkpoint_dir: Path,
    resolution: int,
    precision_mode: str,
    seed: int,
) -> tuple[dict, dict[str, float]]:
    heldout_assets = [asset for asset in manifest.assets if asset.split_name == "test_real_heldout"]
    heldout_manifest = manifest.model_copy(update={"assets": heldout_assets})
    baseline_settings = BaselineRuntimeConfig(
        model_variant=str(checkpoint_dir),
        resolution=resolution,
        precision_mode=precision_mode,
        batch_size=1,
        seed=seed,
    )
    runner = Florence2Runner(baseline_settings=baseline_settings, manifest_base_dir=manifest_base_dir)
    result = runner.run(heldout_manifest)
    if result.status != "completed":
        raise RuntimeError(result.notes or "Florence-2 evaluation failed to complete.")

    metrics, _ = compute_detection_metrics(heldout_manifest, result.predictions)
    summary = {
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        # The local metric implementation computes IoU@0.5 AP-style scores.
        "mAP50": float(metrics.get("mAP", 0.0)),
        "mAP50_95": float(metrics.get("mAP", 0.0)),
    }
    per_class = {str(key): float(value) for key, value in (metrics.get("per_class_ap", {}) or {}).items()}
    return summary, per_class


def train_florence2(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    gradient_accumulation_steps: int = 8,
    resolution: int = 640,
    precision_mode: str = "fp16",
    seed: int = 42,
    model_source: str = "microsoft/Florence-2-base",
) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    manifest_file = Path(manifest_path).resolve()
    manifest_base_dir = manifest_file.parent
    manifest = BenchmarkManifest.model_validate(read_json(manifest_file))
    report_time = datetime.now(timezone.utc).isoformat()

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    run_dir = output_path / f"florence2-fulltrain-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    best_checkpoint_dir = run_dir / "best"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_samples = _build_samples(manifest, manifest_base_dir, "train")
    val_samples = _build_samples(manifest, manifest_base_dir, "val")
    if not train_samples:
        raise ValueError("No train assets with resolvable image paths were found in the manifest.")

    class_name_by_id = {
        entry.class_id: entry.canonical_name
        for entry in manifest.classes
        if entry.status == "active"
    }

    model_repo = model_source
    config = AutoConfig.from_pretrained(model_repo, trust_remote_code=True)
    for candidate in (config, getattr(config, "text_config", None), getattr(config, "language_config", None)):
        if candidate is not None and not hasattr(candidate, "forced_bos_token_id"):
            setattr(candidate, "forced_bos_token_id", None)
    processor = AutoProcessor.from_pretrained(model_repo, trust_remote_code=True, config=config)
    # Keep master weights in fp32; autocast handles fp16 compute safely during training.
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        trust_remote_code=True,
        torch_dtype=dtype,
        config=config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and precision_mode.lower() == "fp16")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    best_val_loss = float("inf")
    steps = 0
    train_losses: list[float] = []
    val_history: list[float] = []
    started_at = time.perf_counter()
    log_every_steps = 100

    print(
        (
            f"[train-florence2] starting run_dir={run_dir} "
            f"device={device.type} epochs={epochs} "
            f"train_assets={len(train_samples)} val_assets={len(val_samples)} "
            f"grad_accum={gradient_accumulation_steps}"
        ),
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        epoch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for index, sample in enumerate(train_samples, start=1):
            with Image.open(sample.image_path).convert("RGB") as image:
                target = _build_detection_target(sample.asset, class_name_by_id)
                target_text = target if target else "<loc_0><loc_0><loc_0><loc_0>background"
                model_inputs = processor(text=_OD_PROMPT, images=image, return_tensors="pt")
                labels = processor.tokenizer(
                    target_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )["input_ids"]
            model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda" and precision_mode.lower() == "fp16"):
                outputs = model(
                    input_ids=model_inputs["input_ids"],
                    pixel_values=model_inputs.get("pixel_values"),
                    labels=labels,
                )
                loss = outputs.loss / max(gradient_accumulation_steps, 1)

            scaled_loss = float(loss.item() * max(gradient_accumulation_steps, 1))
            train_losses.append(scaled_loss)
            epoch_losses.append(scaled_loss)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if index % gradient_accumulation_steps == 0 or index == len(train_samples):
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if steps % log_every_steps == 0:
                    avg_recent = sum(train_losses[-log_every_steps:]) / min(len(train_losses), log_every_steps)
                    elapsed = time.perf_counter() - started_at
                    print(
                        (
                            f"[train-florence2] epoch={epoch}/{epochs} "
                            f"sample={index}/{len(train_samples)} step={steps} "
                            f"loss={scaled_loss:.4f} avg_recent_loss={avg_recent:.4f} "
                            f"elapsed_s={elapsed:.1f}"
                        ),
                        flush=True,
                    )

        # Validation pass for best-checkpoint selection.
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for sample in val_samples:
                with Image.open(sample.image_path).convert("RGB") as image:
                    target = _build_detection_target(sample.asset, class_name_by_id)
                    target_text = target if target else "<loc_0><loc_0><loc_0><loc_0>background"
                    model_inputs = processor(text=_OD_PROMPT, images=image, return_tensors="pt")
                    labels = processor.tokenizer(
                        target_text,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["input_ids"]
                model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
                labels = labels.to(device)
                outputs = model(
                    input_ids=model_inputs["input_ids"],
                    pixel_values=model_inputs.get("pixel_values"),
                    labels=labels,
                )
                val_losses.append(float(outputs.loss.item()))
        mean_val_loss = float(sum(val_losses) / len(val_losses)) if val_losses else float("inf")
        val_history.append(mean_val_loss)
        epoch_elapsed = time.perf_counter() - epoch_start
        mean_train_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else float("inf")
        print(
            (
                f"[train-florence2] epoch_complete epoch={epoch}/{epochs} "
                f"train_loss={mean_train_loss:.4f} val_loss={mean_val_loss:.4f} "
                f"epoch_elapsed_s={epoch_elapsed:.1f}"
            ),
            flush=True,
        )
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            model.save_pretrained(best_checkpoint_dir)
            processor.save_pretrained(best_checkpoint_dir)
            print(
                f"[train-florence2] new_best_checkpoint path={best_checkpoint_dir} val_loss={best_val_loss:.4f}",
                flush=True,
            )
        model.train()

    if not best_checkpoint_dir.exists():
        model.save_pretrained(best_checkpoint_dir)
        processor.save_pretrained(best_checkpoint_dir)

    summary_metrics, per_class = _evaluate_checkpoint(
        manifest=manifest,
        manifest_base_dir=manifest_base_dir,
        checkpoint_dir=best_checkpoint_dir,
        resolution=resolution,
        precision_mode=precision_mode,
        seed=seed,
    )
    print(
        (
            f"[train-florence2] evaluation_complete "
            f"mAP50={summary_metrics['mAP50']:.4f} "
            f"precision={summary_metrics['precision']:.4f} "
            f"recall={summary_metrics['recall']:.4f}"
        ),
        flush=True,
    )

    return {
        "report_type": "florence2_fulltrain_evaluation",
        "generated_at": report_time,
        "checkpoint": str(best_checkpoint_dir),
        "base_model": model_repo,
        "manifest": str(manifest_file),
        "split": "test_real_heldout",
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "resolution": resolution,
            "precision_mode": precision_mode,
            "seed": seed,
            "train_assets": len(train_samples),
            "val_assets": len(val_samples),
            "optimizer": "AdamW",
            "optimization_steps": steps,
            "avg_train_loss": float(sum(train_losses) / len(train_losses)) if train_losses else None,
            "best_val_loss": None if best_val_loss == float("inf") else best_val_loss,
            "val_loss_history": val_history,
        },
        "summary_metrics": summary_metrics,
        "per_class_mAP50_95": per_class,
    }


def train_florence2_command(
    manifest: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=False),
    output_dir: Path = typer.Option(Path("runs/vlm/artifacts/training")),
    epochs: int = typer.Option(1),
    learning_rate: float = typer.Option(1e-5),
    gradient_accumulation_steps: int = typer.Option(8),
    resolution: int = typer.Option(640),
    precision_mode: str = typer.Option("fp16"),
    seed: int = typer.Option(42),
    model_source: str = typer.Option("microsoft/Florence-2-base"),
    report: Path = typer.Option(Path("artifacts/reports/florence2-best-eval.json")),
) -> None:
    payload = train_florence2(
        manifest_path=manifest,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        resolution=resolution,
        precision_mode=precision_mode,
        seed=seed,
        model_source=model_source,
    )
    write_json(report, payload)
    typer.echo(f"Wrote Florence-2 full-training report to {report}")

