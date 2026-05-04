from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.manifests.models import BenchmarkManifest, DatasetAnnotation, DatasetAsset
from src.data.manifests.validator import validate_benchmark_manifest
from src.data.ontology.models import HouseholdObjectClass
from src.utils.paths import write_json


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratified_split_rows(
    labels: list[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[int, str]:
    """Map dataframe row index -> split_name (stratified by label)."""
    rng = random.Random(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_class[label].append(index)
    assignment: dict[int, str] = {}
    for indices in by_class.values():
        rng.shuffle(indices)
        n = len(indices)
        if n == 1:
            assignment[indices[0]] = "train"
            continue
        if n == 2:
            assignment[indices[0]] = "train"
            assignment[indices[1]] = "val"
            continue
        n_train = max(1, min(n - 2, int(round(train_ratio * n))))
        n_val = max(1, min(n - n_train - 1, int(round(val_ratio * n))))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_val = max(1, n - n_train - n_test)
        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]
        for i in train_idx:
            assignment[i] = "train"
        for i in val_idx:
            assignment[i] = "val"
        for i in test_idx:
            assignment[i] = "test_real_heldout"
    return assignment


def build_kitchenware_manifest(
    dataset_root: Path,
    *,
    manifest_id: str = "kitchenware-kaggle",
    seed: int = 42,
    train_ratio: float = 0.75,
    val_ratio: float = 0.125,
    max_rows: int | None = None,
    output_path: Path | None = None,
) -> BenchmarkManifest:
    """
    Build a benchmark manifest from the Kaggle Kitchenware Classification layout:
    ``dataset_root/train.csv`` (columns ``Id``, ``label``) and ``dataset_root/images/{Id}.jpg``.
    Each image gets one full-frame bounding box for open-vocabulary detection training (Florence-2 `<OD>`).
    """
    root = dataset_root.resolve()
    csv_path = root / "train.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing {csv_path} (expected Kaggle competition layout).")
    frame = pd.read_csv(csv_path, dtype={"Id": str})
    if "Id" not in frame.columns or "label" not in frame.columns:
        raise ValueError("train.csv must contain columns Id and label.")
    frame = frame.dropna(subset=["Id", "label"])
    if max_rows is not None:
        frame = frame.head(int(max_rows)).copy()

    labels = [str(row.label).strip().lower() for row in frame.itertuples(index=False)]
    class_names = sorted(set(labels))
    if not class_names:
        raise ValueError("No labeled rows after reading train.csv.")

    split_by_index = _stratified_split_rows(labels, seed=seed, train_ratio=train_ratio, val_ratio=val_ratio)

    classes = [
        HouseholdObjectClass(class_id=name, canonical_name=name, aliases=[], status="active")
        for name in class_names
    ]

    assets: list[DatasetAsset] = []
    counts = {"train": 0, "val": 0, "test_real_heldout": 0}
    by_class_counts: dict[str, int] = {name: 0 for name in class_names}

    for row_index, row in enumerate(frame.itertuples(index=False)):
        image_id = str(row.Id).strip()
        label = str(row.label).strip().lower()
        image_path = root / "images" / f"{image_id}.jpg"
        if not image_path.is_file():
            continue
        with Image.open(image_path).convert("RGB") as image:
            width, height = image.size
        split_name = split_by_index.get(row_index, "train")
        counts[split_name] += 1
        by_class_counts[label] = by_class_counts.get(label, 0) + 1

        rel = str(image_path.resolve())
        assets.append(
            DatasetAsset(
                asset_id=f"kitchenware-{image_id}",
                source_id="kaggle_kitchenware",
                original_identifier=image_id,
                relative_path=rel,
                width=width,
                height=height,
                split_name=split_name,  # type: ignore[arg-type]
                content_hash=_file_sha256(image_path),
                review_status="accepted",
                annotations=[
                    DatasetAnnotation(
                        annotation_id=f"kitchenware-{image_id}-full",
                        class_id=label,
                        source_label=label,
                        bbox_xyxy=[0.0, 0.0, float(width), float(height)],
                        is_ignored=False,
                    )
                ],
            )
        )

    if not assets:
        raise ValueError(f"No images found under {root / 'images'} matching train.csv rows.")

    manifest = BenchmarkManifest(
        manifest_id=manifest_id,
        ontology_version="kitchenware-v1",
        source_ids=["kaggle_kitchenware"],
        split_versions={
            "train": f"kitchenware-train-{seed}",
            "val": f"kitchenware-val-{seed}",
            "test_real_heldout": f"kitchenware-test-{seed}",
        },
        asset_counts={
            "train": counts["train"],
            "val": counts["val"],
            "test_real_heldout": counts["test_real_heldout"],
            "by_class": by_class_counts,
        },
        created_at=datetime.now(timezone.utc).isoformat(),
        classes=classes,
        assets=assets,
    )

    payload = manifest.model_dump(mode="json", exclude_none=True)
    validate_benchmark_manifest(payload)
    if output_path is not None:
        write_json(output_path, payload)
    return manifest
