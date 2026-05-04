# Kitchenware (Florence-2) vs COCO Phase-1 (YOLO11) — Evaluation Report

**Generated:** 2026-05-04  
**Purpose:** Document the Kaggle Kitchenware workflow, per-class results, and a controlled comparison to the repository’s best YOLO11 run on the Phase 1 COCO-derived benchmark.

---

## Part 1 — What we did with the Kitchenware dataset

### Data source

- **Competition:** [Kaggle Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification) (images + `train.csv` with columns `Id`, `label`).
- **Local path used:** `…\assets\kitchenware-classification\` with `train.csv` and `images\{Id}.jpg`.
- **Classes (6):** `cup`, `fork`, `glass`, `knife`, `plate`, `spoon`.

### Adaptation for this codebase

The competition is **image-level classification**, but our Florence-2 training path is the **open-vocabulary detection** task (`<OD>`), which expects **bounding boxes** in training targets.

- We built a **benchmark manifest** (`artifacts/manifests/kitchenware-kaggle-full.json`) where each image has **one tight supervision box covering the full image** `[0, 0, width, height]` and the class label from `train.csv`.
- That lets the existing `train-florence2` pipeline teach the model to emit `<loc_*>` tokens plus the class name for each image, without manual box annotation.

### Splits

- **Stratified** assignment per class into `train` / `val` / `test_real_heldout` (seed 42; ~75% / 12.5% / 12.5% style split per class).
- **Assets in manifest:** 5,559 labeled images with resolvable files (full competition train split used for manifest construction).

### Model and training

- **Model:** `microsoft/Florence-2-base` (fine-tuning), detection head via `<OD>` token supervision.
- **Run:** 1 epoch, AdamW, LR `1e-5`, gradient accumulation 8, FP16 on CUDA, resolution 640 (training loop default).
- **Checkpoint:** `runs/vlm/kitchenware-kaggle-full/florence2-fulltrain-20260504-012243/best`
- **Evaluation:** Same **held-out** split as training reports; metrics from `compute_detection_metrics` (IoU-style AP at 0.5 in this pipeline).

**Authoritative JSON:** `artifacts/reports/kitchenware-kaggle-florence2-full.json`

---

## Part 2 — Kitchenware results (aggregate + per class)

### Aggregate (held-out `test_real_heldout`)

| Metric    | Value   |
| --------- | ------- |
| Precision | 0.409   |
| Recall    | 0.612   |
| mAP50     | 0.368   |

**Training snapshot:** 4,169 train images, 696 val images; ~15.3 min GPU time for one epoch (machine-dependent).

### Per-class mAP50 (held-out)

| Class  | mAP50  |
| ------ | ------ |
| cup    | 0.825  |
| plate  | 0.524  |
| glass  | 0.483  |
| knife  | 0.156  |
| fork   | 0.129  |
| spoon  | 0.093  |

**Spread:** Strongest on **cup** and **plate**; weakest on **spoon** and **fork** (with **knife** only slightly higher). That pattern often appears when objects are small, occluded, or visually similar to the background or to other utensils.

---

## Part 3 — Best YOLO11 results on the Phase 1 COCO dataset (repository baseline)

This is **not** raw full COCO—Phase 1 uses a **curated manifest** from COCO (ontology: **mug** + **book**), exported to a YOLO dataset view and trained end-to-end.

**Authoritative JSON:** `artifacts/reports/yolo11-best-eval.json`  
**Brief write-up:** `artifacts/reports/phase1-summary-brief.md`

### Aggregate (held-out `test_real_heldout`)

| Metric     | Value  |
| ---------- | ------ |
| Precision  | 0.491  |
| Recall     | 0.470  |
| mAP50      | 0.453  |
| mAP50-95   | 0.303  |

### Per-class mAP50-95 (YOLO report field)

| Class | mAP50-95 |
| ----- | -------- |
| mug   | 0.478    |
| book  | 0.128    |

**Training context:** YOLO11 full training run (100 epochs per brief), dedicated detector architecture on **real COCO boxes**, two-class problem.

---

## Part 4 — Comparison: which “performed better,” and why

### Headline numbers (same *kind* of metric family: detection AP on held-out real images)

| Run                         | Dataset / task                         | mAP50 (summary) | Precision | Recall  |
| --------------------------- | -------------------------------------- | ---------------- | --------- | ------- |
| **YOLO11 (Phase 1 COCO)**   | COCO subset, **2 classes**, real boxes | **0.453**        | **0.491** | 0.470   |
| **Florence-2 (Kitchenware)** | Kaggle kitchenware, **6 classes**, full-frame pseudo-boxes | 0.368 | 0.409 | **0.612** |

On **overall mAP50** and **precision**, the **YOLO11 COCO-phase1** run is higher. On **recall**, **Florence-2 on kitchenware** is higher.

### Why a naive “winner” is misleading

1. **Different problems:** Six kitchen utensil classes vs two household COCO classes (**mug**, **book**) — difficulty and ambiguity differ; class count alone changes mAP.
2. **Supervision quality:** COCO YOLO training uses **real instance boxes**. Kitchenware Florence-2 training used **full-image boxes** for every object — a standard trick for single-object classification images, but it **does not teach fine localization** the way COCO boxes do. That caps how meaningful IoU-based AP is for “detection quality.”
3. **Model roles:** **YOLO** is a **pure detector** optimized for speed and AP on fixed classes. **Florence-2** here was **fine-tuned as a detector-style output** on one epoch; it is not the same as a long-trained specialist detector on the same label space.
4. **Metric alignment:** YOLO’s report includes **mAP50-95** (stricter); Florence-2 evaluation in this repo’s report duplicates mAP50 for the 50–95 slot (implementation detail in `train_florence2` evaluation). Prefer **mAP50** when comparing these two JSON files directly.

### Plain-language conclusion

- **For overall detection AP on each project’s own held-out split:** **YOLO11 on the Phase 1 COCO benchmark scored higher mAP50** than **Florence-2 on the adapted kitchenware setup**, mainly reflecting **real bounding-box supervision** and a **detector specialized for that two-class task**.
- **Kitchenware Florence-2** still shows **strong per-class signal on some classes (e.g. cup)** and **higher recall** in aggregate, which can happen when the model fires more predictions — useful for coverage, but not the same as winning on strict AP with tight boxes.

### If you want a fairer future comparison

- Train **YOLO** on the **same kitchenware manifest** (same splits, same full-frame boxes), or  
- Add **real / tight boxes** for kitchenware, or  
- Compare **classification accuracy** on Kaggle labels for models suited to classification.

---

## References (paths in this repo)

| Artifact | Path |
| -------- | ---- |
| Kitchenware Florence-2 full report | `artifacts/reports/kitchenware-kaggle-florence2-full.json` |
| Kitchenware manifest (full) | `artifacts/manifests/kitchenware-kaggle-full.json` |
| YOLO11 best eval (COCO phase 1) | `artifacts/reports/yolo11-best-eval.json` |
| Phase 1 YOLO brief | `artifacts/reports/phase1-summary-brief.md` |
