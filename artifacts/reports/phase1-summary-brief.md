# Phase 1 Brief: YOLO11 Full Training Outcome

## Final Status

- Full YOLO11 training completed successfully on GPU (`CUDA`, `100` epochs).
- Authoritative model for Phase 1 is the training **best checkpoint**, not final epoch weights.
- Held-out evaluation was run on `test_real_heldout`.

## Best-Checkpoint Metrics (Held-Out)

- Model checkpoint: `runs/detect/artifacts/training/yolo11-fulltrain-gpu/weights/best.pt`
- Precision: `0.4908`
- Recall: `0.4704`
- mAP50: `0.4527`
- mAP50-95: `0.3030`

## Class-Level Result

- `mug` mAP50-95: `0.4777`
- `book` mAP50-95: `0.1284`

## Interpretation

- YOLO11 improves substantially versus the earlier quick baseline and is now the strongest implemented path.
- Performance is uneven by class; `book` detection remains the main bottleneck.
- Training peaked mid-run, so checkpoint selection is critical for stable deployment quality.

## Recommended Next Actions

1. Use `best.pt` for all inference/evaluation workflows.
2. Prioritize `book` improvements (class balancing, hard examples, targeted augmentations).
3. Re-run cross-model comparison once Florence-2 and Grounding DINO live inference paths are unblocked.

## Traceability

- Detailed progress summary: `artifacts/reports/phase1-summary.json`
- Best-checkpoint evaluation report: `artifacts/reports/yolo11-best-eval.json`
