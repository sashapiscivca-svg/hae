# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/). Model files are versioned together with the code: a change to any released weight is a new minor version at least.

## [1.0.0] — 2026-10-15

First public release.

### Model
- `hae-s-ir`: thermal person detector, 313 k parameters (recipe ir-v3: 90 epochs from scratch, HIT-UAV + AIResQ, scale jitter 0.9–2.4, inference ×1.5).
- ONNX fp32, ONNX int8 (fp32 head), PyTorch safetensors, `config.json`, `metrics.json`, SHA-256 sums.

### Data
- Converted annotations of all 5 training and evaluation splits (`data/annotations/`), data card, conversion pipeline.

### Code
- Runtime package without PyTorch: `Detector`, numpy decoding, SAHI-style tiling, CLI (`hae models|info|predict|export|download`).
- Training, export with int8 calibration, evaluation (AP50, recall / precision / F1 / false alarms per frame) and benchmarking scripts.
