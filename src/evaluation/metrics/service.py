from __future__ import annotations

from collections import defaultdict
from statistics import mean

from src.data.manifests.models import BenchmarkManifest, FailureExample
from src.models.common.predictions import AssetPrediction


def _iou(box_a: list[float], box_b: list[float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max(box_a[2] - box_a[0], 0) * max(box_a[3] - box_a[1], 0)
    area_b = max(box_b[2] - box_b[0], 0) * max(box_b[3] - box_b[1], 0)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def compute_detection_metrics(manifest: BenchmarkManifest, predictions: list[AssetPrediction]) -> tuple[dict, list[FailureExample]]:
    prediction_map = {item.asset_id: item.predictions for item in predictions}
    stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    failures: list[FailureExample] = []

    for asset in manifest.assets:
        gt_annotations = [annotation for annotation in asset.annotations if not annotation.is_ignored]
        predicted_annotations = prediction_map.get(asset.asset_id, [])
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()

        for pred_index, prediction in enumerate(predicted_annotations):
            best_gt = None
            best_iou = 0.0
            for gt_index, ground_truth in enumerate(gt_annotations):
                if gt_index in matched_gt or prediction.class_id != ground_truth.class_id:
                    continue
                iou_value = _iou(prediction.bbox_xyxy, ground_truth.bbox_xyxy)
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt = gt_index
            if best_gt is not None and best_iou >= 0.5:
                matched_gt.add(best_gt)
                matched_pred.add(pred_index)
                stats[prediction.class_id]["tp"] += 1

        for pred_index, prediction in enumerate(predicted_annotations):
            if pred_index in matched_pred:
                continue
            stats[prediction.class_id]["fp"] += 1
            failures.append(
                FailureExample(
                    failure_id=f"{asset.asset_id}-fp-{pred_index}",
                    run_id="pending",
                    asset_id=asset.asset_id,
                    failure_type="background_confusion",
                    predicted_class_id=prediction.class_id,
                )
            )

        for gt_index, ground_truth in enumerate(gt_annotations):
            if gt_index in matched_gt:
                continue
            stats[ground_truth.class_id]["fn"] += 1
            failures.append(
                FailureExample(
                    failure_id=f"{asset.asset_id}-fn-{gt_index}",
                    run_id="pending",
                    asset_id=asset.asset_id,
                    failure_type="occlusion",
                    expected_class_id=ground_truth.class_id,
                )
            )

    per_class_ap: dict[str, float] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    for class_id, values in stats.items():
        tp = values["tp"]
        fp = values["fp"]
        fn = values["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        ap = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        per_class_ap[class_id] = round(ap, 6)
        precisions.append(precision)
        recalls.append(recall)

    metrics = {
        "mAP": round(mean(per_class_ap.values()) if per_class_ap else 0.0, 6),
        "precision": round(mean(precisions) if precisions else 0.0, 6),
        "recall": round(mean(recalls) if recalls else 0.0, 6),
        "per_class_ap": per_class_ap,
    }
    return metrics, failures

