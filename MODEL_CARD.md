# Model card: hae-s-ir v1.0.0

[Українська](MODEL_CARD.uk.md)

This card follows the structure of *Model Cards for Model Reporting* (Mitchell et al., 2019). Machine-readable numbers are in [`models/hae-s-ir/metrics.json`](models/hae-s-ir/metrics.json), and the full recipe is in [`models/hae-s-ir/config.json`](models/hae-s-ir/config.json).

## 1. Model details

| | |
|---|---|
| Developed by | The hae authors |
| Release | v1.0.0, 2026-10-15 |
| Model type | Single-stage, anchor-free, dense object detector, one class (`person`), output stride 4 |
| Input | Thermal (LWIR) 8-bit grayscale frame, 1 channel, per-frame z-score, recommended upscale ×1.5 |
| Size | 313,169 parameters; 0.269 GMAC per 256×320 tile (×1.0); 383 KB int8 ONNX, 1.2 MB fp32 ONNX |
| Formats | ONNX opset 17 (fp32; int8 with fp32 head); PyTorch `safetensors` (training graph, unfused) |
| Initialisation | Random. No pretrained or third-party weights. |
| Training compute | One NVIDIA T4, 42 min (90 epochs) |
| Licence | MIT |
| Citation | see [README](README.md#citation) |
| Contact | GitHub issues; security and safety reports: [SECURITY.md](SECURITY.md) |

**Architecture.** The stem is space-to-depth followed by a 3×3 convolution (stride 4). The backbone has Ghost-block stages at strides 4/8/16 (widths 40/56/80) and a cheap stride-32 context level. A top-down neck (40 channels) feeds a stride-4 head that predicts one logit and four box-side distances per cell. Decoding takes 3×3 local maxima of the sigmoid score and applies softplus(distance) × 4 from the cell centre. No non-maximum suppression is needed. Details: [docs/en/architecture.md](docs/en/architecture.md).

**Training.** Loss: Quality Focal Loss with an IoU target, plus GIoU and NWD box losses. Ignore regions are not penalised. The negative loss is down-weighted close to people ("soft negatives"), which makes the model tolerant of imprecise box centres in the labels. Optimiser: AdamW with lr 3e-3, weight decay 0.01, 50 warm-up steps and cosine decay; the exported weights are an EMA (0.998). Batch 32 random 256×320 crops, 4,900 crops per epoch, 90 epochs. Crops come 60 % from HIT-UAV trainval and 40 % from AIResQ train. Augmentation: scale jitter 0.9–2.4 and copy-paste of people (p = 0.3).

**Quantisation.** Static int8 (onnxruntime; QDQ format, uint8 activations, per-channel int8 weights), calibrated on 256 training tiles at ×1.5. The final 1×1 classification and box convolutions stay in fp32. AP50 changes by −0.008 to +0.001 relative to fp32.

## 2. Intended use

**Primary use.** Decision support for civilian search and rescue: flagging likely people in thermal imagery from drones flying roughly 30–120 m above ground. Examples are floods, earthquakes, open-water and coastal searches, and missing persons in forests or mountains. The detector is meant to run on board a cheap companion computer (Raspberry Pi Zero 2 W class) or on a ground station, and to feed a human operator or a tracking stage.

**Primary users.** Rescue teams, volunteer SAR groups, researchers and developers of open rescue tooling.

**Out of scope.** Any use that targets people for harm, including weapons, targeting, or surveillance and tracking of individuals or groups without their consent. Identifying who a person is (the model cannot do this and must not be extended to do it). Sole-source, life-critical decisions; in particular, calling off a search because the model found nothing. RGB imagery, ground-level, indoor or close-range imagery. See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md).

## 3. Factors

Performance changes strongly with:

- **Target size in pixels.** This depends on altitude, focal length and sensor resolution. The training data has median person heights of 19 px (HIT-UAV) and 8–9 px (AIResQ) in 640×512 frames. With the default ×1.5 upscale, the model is tuned for small people. Larger targets (median 32 px in WiSARD-IR) are detected worse. A lower scale (×1.0) helps there.
- **Sensor and thermal contrast.** This depends on the camera model, its auto-gain, ambient temperature, sun loading and clothing. Performance does not transfer reliably across cameras.
- **Scene.** Vegetation cover, water, rubble, crowds and occlusion.

We have not measured performance across demographic groups. At these resolutions the model sees warm blobs a few pixels tall, and the datasets carry no such labels.

## 4. Metrics

- **AP50.** COCO-style average precision at IoU 0.5.
- **Operator metrics** at a fixed threshold: recall, precision, F1 and **false alarms per frame**. Two presets are stored in the ONNX file. `balanced` (0.50) maximises mean F1 and `sensitive` (0.45) maximises mean F2. Both were chosen on HIT-UAV test and AIResQ val.
- Matching is greedy and class-agnostic at IoU ≥ 0.5. Detections inside ignore regions are not counted as false alarms.
- **Uncertainty.** The model comes from one training run. Repeated runs of similar recipes varied by about 0.02 AP50 on HIT-UAV and 0.04–0.08 on AIResQ. No confidence intervals are given.

## 5. Evaluation data

| Split | Source | Frames | People | Median height | Role |
|---|---|---:|---:|---:|---|
| `hituav_test` | HIT-UAV official test split | 579 | 2,611 | 19 px | in-domain test |
| `airesq_val` | AIResQ, ~14 % of recording groups held out whole | 502 | 933 | 8 px | in-domain validation |
| `wisard_ir_val` | WiSARD thermal, sites Karen/MtErie/Airfield | 1,633 | 2,439 | 32 px | **transfer test**: never trained or tuned on |

No evaluation split shares a scene with training. The annotations are in `data/annotations/`.

**Caveat.** The presets, the int8 calibration choice and the recipe were selected using `hituav_test` and `airesq_val`, so the results on those two sets are slightly optimistic. `wisard_ir_val` is untouched by any choice.

## 6. Training data

`hituav_trainval` (2,319 frames, 9,701 people; 60 % of crops) and `airesq_train` (1,486 frames, 1,710 people; 40 %). Both are civilian datasets, released under CC0 1.0 and CC BY 4.0. Details: [DATA_CARD.md](DATA_CARD.md).

## 7. Quantitative analysis

| Set | AP50 fp32 | AP50 int8 | Preset | Recall | Precision | F1 | FP per frame |
|---|---:|---:|---|---:|---:|---:|---:|
| HIT-UAV test | 0.758 | 0.757 | balanced 0.50 | 0.728 | 0.750 | 0.739 | 1.09 |
| | | | sensitive 0.45 | 0.773 | 0.700 | 0.735 | 1.50 |
| AIResQ val | 0.238 | 0.230 | balanced 0.50 | 0.521 | 0.238 | 0.326 | 3.11 |
| | | | sensitive 0.45 | 0.564 | 0.191 | 0.286 | 4.42 |
| WiSARD-IR val | 0.086 | 0.087 | balanced 0.50 | 0.166 | 0.162 | 0.164 | 1.28 |
| | | | sensitive 0.45 | 0.188 | 0.131 | 0.154 | 1.86 |

All operator metrics are for int8 at ×1.5. Sweeps over 12 thresholds (0.05–0.60) for fp32 and int8 are in `metrics.json`.

**Comparison with the other sizes we trained (same recipe, not released).** On HIT-UAV test / AIResQ val (int8 AP50): n 0.633 / 0.255 (53 k parameters), t 0.691 / 0.214 (118 k), **s 0.757 / 0.230 (313 k)**, m 0.739 / 0.273 (821 k). s is the best on HIT-UAV, and it is the largest size within the Pi Zero 2 W budget (≈0.3 GMAC per tile).

**Analysis of the transfer gap.** Running at ×1.0 instead of ×1.5 improved WiSARD-IR AP50 by 0.01–0.05 across the sizes we trained, but lowered HIT-UAV and AIResQ by 0.03–0.12. Test-time contrast normalisation and BatchNorm re-estimation (AdaBN) did not close the gap. Training on more diverse thermal data is the open problem.

**Latency** on a Raspberry Pi Zero 2 W (int8): 148 ms per 256×320 tile on 1 thread, 55 ms on 4 threads; a 640×512 frame at ×1.5 takes 554 ms on 4 threads. See the [README](README.md#speed-on-raspberry-pi-zero-2-w).

## 8. Ethical considerations

- **Missed people cost lives; false alarms cost time.** Neither error rate is low enough for autonomous decisions. Always keep a human in the loop and never treat "no detection" as "no person".
- **Dual use.** A person detector for aerial imagery could be misused for surveillance or targeting. We release it because the life-saving value for under-resourced rescue teams is high and the capability is already widely available. Our [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) states the uses we ask people to refrain from. The MIT licence itself does not restrict use.
- **Privacy.** The model outputs boxes, not identities. Drone footage of people is still personal data in many jurisdictions. Follow local law and SAR protocols on recording and retention.

## 9. Caveats and recommendations

1. Validate on footage from **your** camera, altitude and terrain before an operation. `scripts/evaluate.py` works on any COCO-annotated set.
2. Fine-tune from `hae-s-ir.safetensors` if you have even a few hundred labelled frames from your sensor.
3. Pick the threshold for your tolerance of false alarms. `sensitive` finds more people but raises 1.5–4.4 false alarms per frame on the evaluation sets.
4. Confirm detections over time, with a tracker or by requiring persistence across frames or passes, before alerting.
5. Use ×1.0 when people appear larger than about 25 px, and keep ×1.5 for small, distant people.
6. On a Pi Zero 2 W, fit a heatsink, use the int8 model, and run the detector every K frames with tracking in between.
