from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.manifests.kitchenware import build_kitchenware_manifest


def test_build_kitchenware_manifest_full_frame_boxes(tmp_path: Path) -> None:
    root = tmp_path / "kw"
    (root / "images").mkdir(parents=True)
    df = pd.DataFrame([{"Id": "0001", "label": "cup"}, {"Id": "0002", "label": "plate"}])
    df.to_csv(root / "train.csv", index=False)
    for rid, size in (("0001", (80, 60)), ("0002", (64, 64))):
        Image.new("RGB", size, color=(10, 20, 30)).save(root / "images" / f"{rid}.jpg")

    manifest = build_kitchenware_manifest(root, manifest_id="kw-test", seed=1)
    assert len(manifest.assets) == 2
    assert {c.class_id for c in manifest.classes} == {"cup", "plate"}
    for asset in manifest.assets:
        assert len(asset.annotations) == 1
        ann = asset.annotations[0]
        assert ann.bbox_xyxy == [0.0, 0.0, float(asset.width), float(asset.height)]
