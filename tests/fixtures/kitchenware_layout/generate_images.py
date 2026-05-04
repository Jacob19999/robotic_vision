"""Create JPEGs listed in train.csv (same layout as Kaggle kitchenware-classification)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    frame = pd.read_csv(root / "train.csv", dtype={"Id": str})
    images = root / "images"
    images.mkdir(exist_ok=True)
    for row in frame.itertuples(index=False):
        rgb = ((hash(row.Id) % 200) + 20, (hash(row.label) % 200) + 20, 80)
        Image.new("RGB", (128, 96), rgb).save(images / f"{row.Id}.jpg", quality=85)
    print(f"Wrote {len(frame)} images to {images}")
