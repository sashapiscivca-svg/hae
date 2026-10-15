# Evaluation

[Українська](../uk/evaluation.md) · [Docs index](README.md)

## Protocol used for the published numbers

- **Sets:** `hituav_test` (official HIT-UAV test split), `airesq_val` (AIResQ recording groups held out whole) and `wisard_ir_val` (WiSARD thermal, three held-out sites, never used for training or for any choice). The annotations are in `data/annotations/`.
- **Input:** the int8 ONNX model (fp32 is reported alongside it) on full 640×512 frames, upscaled ×1.5, without tiling or test-time augmentation.
- **Matching:** class-agnostic greedy matching of detections, sorted by score, to ground truth at IoU ≥ 0.5. Detections that fall into an ignore region (category 3) are neither true nor false positives.
- **AP50:** COCO-style area under the interpolated precision–recall curve at IoU 0.5, for the `person` class.
- **Operator metrics** at a confidence threshold: recall, precision, F1, and false alarms per frame. The model's two presets come from the ONNX metadata: `balanced` (max mean F1) and `sensitive` (max mean F2), both chosen on HIT-UAV test and AIResQ val.
- **One run.** Each published number comes from one training run. On similar recipes, repeated runs differed by about 0.02 AP50 on HIT-UAV and 0.04–0.08 on AIResQ.

## Run it

```bash
python scripts/evaluate.py --model models/hae-s-ir/hae-s-ir_lwir_int8.onnx --data data --sets hituav_test,airesq_val,wisard_ir_val
python scripts/evaluate.py --model models/hae-s-ir/hae-s-ir_lwir_int8.onnx --data data --sets airesq_val --conf 0.3,0.4,0.5 --json airesq.json
python scripts/evaluate.py --model models/hae-s-ir/hae-s-ir_lwir.onnx --data data --sets wisard_ir_val --scale 1.0
```

The script prints AP50 and the operator metrics per set, and with `--json` it saves the full report. `--every k` evaluates every k-th frame for a quick check. The numbers it produces for the released model should match `models/hae-s-ir/metrics.json`.

## Evaluating on your own footage

This is the most important step before an operation. Annotate a few hundred frames that the model has never seen, from **your** camera at **your** altitudes, in COCO format (see [data.md](data.md#using-your-own-data)), then run `scripts/evaluate.py` on them. Look at:

1. **recall at the threshold you plan to use**: what share of people would the operator be shown?
2. **false alarms per frame**: multiply by the number of frames the detector processes per minute to get the operator's alarm load per minute. Temporal confirmation (tracking) lowers this load a lot.
3. **the `--scale` sweep** (for example 1.0 / 1.25 / 1.5): it can matter more than the threshold when your people are larger or smaller than in the training data.

## Why AP50 underestimates performance on tiny people

For a 6-pixel person, a box that is off by 2 pixels already falls below IoU 0.5 and counts as a miss *and* a false alarm, even though an operator would see the right spot. `hae.recall.recall_center` therefore provides a **centre-based** match: a detection is a hit when its centre lies inside the person's box, enlarged by 2 px, and detections within 8 px of each other count as one alarm. That match is closer to what a rescuer experiences. The published headline numbers use the standard IoU protocol, so they can be compared with the literature.
