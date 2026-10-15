# Inference

[Українська](../uk/inference.md) · [Docs index](README.md)

## Installation

```bash
pip install -e .                 # from a clone: numpy, onnxruntime, opencv-python-headless, onnx, pillow
pip install -e ".[hub]"          # + huggingface_hub, to download models by name
```

## Finding models

`Detector("hae-s-ir")` resolves a model name in this order:

1. an explicit path to an `.onnx` file;
2. the `root=` argument, then the `$HAE_MODELS` directory;
3. `./models` in the current directory, then the `models/` directory of the repository;
4. the Hugging Face Hub (`alexshynkarenk0/hae-s-ir`), if `huggingface_hub` is installed. Files are cached locally.

A model directory is `<root>/<name>/<name>_<mode>[_int8].onnx` with `mode` = `lwir` or `rgb`. The int8 file is the default.

```bash
hae models                         # bundled or local models with their modes and presets
hae info hae-s-ir                  # metadata stored in the ONNX file
hae download hae-s-ir              # fetch from the Hub into the local cache
```

## Python API

```python
from hae.detector import Detector

det = Detector("hae-s-ir")                     # balanced preset, recommended scale
det = Detector("hae-s-ir", preset="sensitive") # higher recall, more false alarms
det = Detector("hae-s-ir", conf=0.3, scale=1.0, threads=4)
det = Detector("hae-s-ir", tile=(256, 320), overlap=0.2)    # tiling for large frames

res = det.predict("frame.png")       # file → Results; directory or list → list[Results]
res.boxes    # (n, 4) x1, y1, x2, y2 in original pixels
res.conf     # (n,) scores
res.cls      # (n,) class ids (0 = person)
res.speed    # {"preprocess": ms, "inference": ms, "postprocess": ms}
res.save("out.png"); res.to_json(); res.summary()

dets = det.detect(numpy_image)       # low level: (n, 6) [x1, y1, x2, y2, score, class], sorted by score
```

Image input: a 2-D `uint8` grayscale array (white-hot thermal). Colour images are converted to grayscale, which is only correct for grayscale-palette thermal frames saved as RGB.

## Command line

```bash
hae predict MODEL SOURCE [--conf C] [--preset balanced|sensitive] [--scale S] [--tile H,W] [--overlap O]
                         [--project runs] [--json] [--no-save]
```

The command writes annotated images (and `results.json` with `--json`) to `runs/predict/exp`, `exp2`, and so on, and prints per-image timing.

## The ONNX contract (for C++, Rust, Java, mobile …)

The model is a single ONNX graph (opset 17) with dynamic batch, height and width. Height and width must be multiples of 32: pad the bottom and right with zeros.

| | `hae-s-ir_lwir.onnx`, `hae-s-ir_lwir_int8.onnx` |
|---|---|
| Input name / shape | `t`: float32 `(B,1,H,W)` |
| Preprocessing | per-frame z-score on the 0..255 scale: `(t − mean(t)) / max(std(t), 1)`, computed on the whole frame **before** scaling or tiling |
| Recommended scale | upscale the normalised frame ×1.5 (bilinear) |
| Outputs | `cls`: `(B,1,H/4,W/4)` logits; `box`: `(B,4,H/4,W/4)` raw distances |

(The code also supports RGB models with an input `rgb`: float32 `(B,3,H,W)`, `pixel / 255`; none is released in v1.0.0.)

Decoding (see `src/hae/decode_np.py`, about 60 lines of numpy):

1. `score = sigmoid(cls)`;
2. keep cells that are 3×3 local maxima of `score` and have `score ≥ conf`, at most the top 300;
3. for a kept cell `(i, j)` with centre `cx = (j + 0.5)·4`, `cy = (i + 0.5)·4`: `l, t, r, b = softplus(box[:, i, j]) · 4`, box = `(cx − l, cy − t, cx + r, cy + b)`;
4. divide the coordinates by the input scale to return to original pixels.

No NMS is needed for a single image. With tiling, merge duplicates across tile borders (`hae.sahi`).

**Metadata** (`onnx.load(path).metadata_props`, JSON-encoded values): `name`, `hae_version`, `mode`, `tnorm`, `stride`, `ncls`, `class_names`, `input_multiple_of`, `recommended_scale`, `recommended_conf`, `sensitive_conf`, `precision`, `preprocess`, `outputs`, `recipe`, `training_data`, `license`, `copyright`, `intended_use`.

## Choosing thresholds

The presets were chosen on validation data: `balanced` maximises F1 and `sensitive` maximises F2. On your own footage, the right threshold depends on how many false alarms per minute your operator can review. `metrics.json` lists recall, precision and false alarms per frame at 12 thresholds (0.05–0.60) for every evaluation set. For a search, we suggest starting with `sensitive` and confirming detections over time.
