---
license: mit
library_name: onnx
pipeline_tag: object-detection
tags:
  - search-and-rescue
  - thermal
  - infrared
  - lwir
  - drone
  - uav
  - tiny-object-detection
  - person-detection
  - edge-ai
  - raspberry-pi
  - onnx
  - int8
datasets:
  - alexshynkarenk0/hae-thermal-sar-annotations
language:
  - en
  - uk
model-index:
  - name: hae-s-ir
    results:
      - task:
          type: object-detection
          name: Person detection (thermal, aerial)
        dataset:
          type: hit-uav
          name: HIT-UAV (test split)
        metrics:
          - type: ap50
            name: AP50 (int8, IoU 0.5)
            value: 0.757
      - task:
          type: object-detection
          name: Person detection (thermal, aerial)
        dataset:
          type: airesq
          name: AIResQ Benchmark (held-out groups)
        metrics:
          - type: ap50
            name: AP50 (int8, IoU 0.5)
            value: 0.230
      - task:
          type: object-detection
          name: Person detection (thermal, aerial; transfer)
        dataset:
          type: wisard
          name: WiSARD thermal (held-out sites, never trained on)
        metrics:
          - type: ap50
            name: AP50 (int8, IoU 0.5)
            value: 0.087
---

# hae-s-ir: open thermal person detector for search-and-rescue drones

![HIT-UAV test frame with hae-s-ir detections: green boxes around 21 of 23 people, orange circles around the 2 missed](assets/hae-hero.jpg)

*A real HIT-UAV test frame (night, 60 m): green — detections with scores, orange — missed people.*

**hae-s-ir** finds people in thermal (LWIR) drone imagery. It has 313 k parameters, the int8 ONNX file is 383 KB, and a 256×320 tile runs in **55 ms on a Raspberry Pi Zero 2 W** (4 threads, measured). It was trained from scratch on openly licensed civilian data, and the code and weights are MIT-licensed.

- **Live demo (runs in your browser):** https://huggingface.co/spaces/alexshynkarenk0/hae-demo
- **Related model:** [hae-n-ir-ext-kdf](https://huggingface.co/alexshynkarenk0/hae-n-ir-ext-kdf) (code: https://github.com/sashapiscivca-svg/hae-n-ir-ext-kdf), a real-time model for forest and wilderness footage
- Code, documentation and training recipe: https://github.com/sashapiscivca-svg/hae
- Full model card (EN): https://github.com/sashapiscivca-svg/hae/blob/main/MODEL_CARD.md · (UK): https://github.com/sashapiscivca-svg/hae/blob/main/MODEL_CARD.uk.md
- Responsible use: https://github.com/sashapiscivca-svg/hae/blob/main/RESPONSIBLE_USE.md

> **Decision support only.** The model misses people and raises false alarms. Never use it as the only means of search, and never call off a search because it found nothing.

## Files

| File | Purpose |
|---|---|
| `hae-s-ir_lwir_int8.onnx` | deployment (int8, fp32 head), 383 KB |
| `hae-s-ir_lwir.onnx` | fp32 reference, 1.2 MB |
| `hae-s-ir.safetensors` | PyTorch weights for fine-tuning with the `hae` package |
| `config.json` | architecture and full training recipe |
| `metrics.json` | all evaluation numbers, including threshold sweeps |

## Usage

```bash
pip install "hae[hub] @ git+https://github.com/sashapiscivca-svg/hae"
hae predict hae-s-ir frames/            # downloads this repository's int8 model on first use
```

```python
from hae.detector import Detector
det = Detector("hae-s-ir")              # preprocessing, ×1.5 scale and threshold from the ONNX metadata
res = det.predict("thermal_frame.png")  # 8-bit grayscale, white-hot
print(res.boxes, res.conf)
```

**Without the hae package:** the input `t` is float32 `(B,1,H,W)` with H and W multiples of 32. Normalise each frame as `(t − mean) / max(std, 1)` on the 0..255 scale, then upscale ×1.5. The outputs are `cls` `(B,1,H/4,W/4)` logits and `box` `(B,4,H/4,W/4)`. Decode by taking 3×3 local maxima of `sigmoid(cls)`; each box is `softplus(box) × 4` around the cell centre `((j+0.5)·4, (i+0.5)·4)`. Then divide by the scale.

## Results (int8, IoU ≥ 0.5)

| Set | AP50 | Recall / precision / false alarms per frame at the balanced preset (0.50) |
|---|---:|---|
| HIT-UAV test (579 frames, 2,611 people) | 0.757 | 0.73 / 0.75 / 1.09 |
| AIResQ val (502 frames, 933 people, ~8 px tall) | 0.230 | 0.52 / 0.24 / 3.11 |
| WiSARD-IR val (transfer; 1,633 frames, 2,439 people) | 0.087 | 0.17 / 0.16 / 1.28 |

The weak transfer to WiSARD-IR (other cameras and sites) is the main limitation. Validate on your own camera and fine-tune if possible.

## Training data and licence

The model was trained on HIT-UAV trainval (CC0 1.0) and AIResQ Benchmark train (CC BY 4.0), and evaluated additionally on WiSARD thermal (MIT). Attribution: Suo et al., *Scientific Data* 10, 227 (2023); Büttner, *AIResQ Benchmark*, Zenodo doi:10.5281/zenodo.17405074 (CC BY 4.0); Broyles, Hayner, Leung, *IROS 2022*. The model weights are released under the MIT License, © 2026 The hae authors.

## Citation

```bibtex
@software{hae_2026,
  title   = {hae: an open thermal person detector for search-and-rescue drones},
  author  = {{The hae authors}},
  year    = {2026},
  version = {1.0.0},
  license = {MIT},
  url     = {https://github.com/sashapiscivca-svg/hae}
}
```
