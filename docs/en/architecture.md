# Architecture

[Українська](../uk/architecture.md) · [Docs index](README.md)

hae is a single-stage, anchor-free, dense detector designed around two constraints: targets that are often **5–30 px** tall, and a compute budget of roughly **0.05–0.3 GMAC per 256×320 tile** on a Raspberry Pi Zero 2 W. Source: `src/hae/model/hae.py`, `blocks.py`, `presets.py`; decoding: `src/hae/decode_np.py`.

## Data flow

```
input (B,1,H,W) thermal frame                        H, W multiples of 32
  │  per-frame normalisation (outside the graph): (t − μ)/max(σ,1) on the 0..255 scale
  ▼
Stem      pixel_unshuffle(2) → 3×3 conv stride 2                     → stride 4
  ▼
s4        1×1 conv → Ghost blocks × d0                               c0 channels, stride 4
s8        Down (3×3 s2) → Ghost blocks × d1                          c1, stride 8
s16       Down → Ghost blocks × d2 → Context (dilated 3×3, residual) c2, stride 16
s32       Down → Ghost block → Context, upsampled and added to s16
  ▼
Neck      top-down: lateral 1×1 + nearest ×2 upsampling, s16 → s8 → s4, 3×3 fuse convs
  ▼
Head      3×3 trunk → cls 1×1 (ncls logits)  and  box 1×1 (4 raw distances)   at stride 4
  ▼
Decode    sigmoid(cls) → 3×3 local maxima → top-K → box = softplus(raw)·stride around the cell centre
          → [x1, y1, x2, y2, score, class]   (no NMS in the graph)
```

**Ghost block**: half the channels come from a 1×1 convolution, the other half from a cheap depthwise 3×3 applied to the first half, with a residual connection. All convolutions are followed by BatchNorm + ReLU, which are folded into the convolution weights at export time.

**Why stride 4 and peaks instead of anchors and NMS.** At stride 4, a 6-pixel person still covers a grid cell. Peak picking on a dense score map replaces anchor matching and NMS, which keeps decoding cheap in numpy on the Pi and avoids the anchor-size tuning that small objects need.

**Why thermal only.** The architecture can also take RGB, or RGB and thermal together (with a per-pixel gate). Separately trained weights per modality scored higher than a joint model in our experiments. The released model therefore has a single thermal input `t`. The RGB line is not part of this release.

## Configuration of hae-s-ir

| Stem | Widths (s4/s8/s16) | Ghost blocks per stage | Neck/head channels | Context s32 | Parameters | GMAC / 256×320 tile |
|---:|---|---|---:|---|---:|---:|
| 24 | 40 / 56 / 80 | 1 / 2 / 2 | 40 | yes | 313,169 | 0.269 |

The code also defines smaller (`n`, `t`) and larger (`m`) presets that share the same design. They were trained with the same recipe but are not part of this release (see the model card for their numbers).

## Training objective

- **Classification:** Quality Focal Loss. The positive target is the IoU between the predicted and the ground-truth box, so the score is calibrated to localisation quality. Positives are the grid cells inside a central region of each box; each one regresses the box it is closest to. **Ignore** regions get no loss.
- **Box:** GIoU plus Normalized Wasserstein Distance (NWD). NWD keeps a useful gradient for boxes of a few pixels, where IoU is 0 for most near-misses.
- **Soft negatives:** the negative loss is multiplied by (1 − G)^β, where G is a Gaussian around each person (σ = max(side/4, stride)) and β = 1. This tolerates the centre noise of hand-drawn tiny boxes.
- **Optimisation:** AdamW (lr 3e-3, weight decay 0.01 on weights), linear warm-up for 50 steps, then cosine decay; EMA of weights (decay 0.998) is what gets exported.
- **Sampling:** random 256×320 crops (batch 32, 4,900 crops per epoch) drawn from several datasets with fixed shares. Scale jitter 0.9–2.4 teaches the model the upscaled views used at inference. Copy-paste of people p = 0.3.

## Inference options

- **Scale.** The model expects the frame upscaled ×1.5 (stored as `recommended_scale`). Cost grows with the square of the scale. Use ×1.0 when people are larger than about 25 px, and ×2 for very small ones.
- **Tiling** (`tile=(256, 320)`): for frames much larger than 640×512. Tiles overlap by 20 % and duplicates are merged (SAHI-style, implemented in numpy).
- **Presets.** `recommended_conf` (balanced, max F1) and `sensitive_conf` (max F2) are stored in each model.
- **Test-time augmentation** (`hae.fusion`): horizontal flip and multi-scale with weighted box fusion. On held-out sets this added +0.005–0.08 AP50 (typically about +0.04), at a multiple of the compute. It is off by default.

## int8 export

Static quantisation with onnxruntime: QDQ format, uint8 activations, per-channel int8 weights, calibrated on 256 training tiles at the deployment scale. The two final 1×1 convolutions (`cls`, `box`) stay in fp32. Quantising them cost up to 0.07 AP50 in our experiments. For hae-s-ir, the int8 model is within 0.008 AP50 of fp32 on all evaluation sets.
