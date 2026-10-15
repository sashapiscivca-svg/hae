# Deploying on a Raspberry Pi Zero 2 W

[Українська](../uk/deployment-raspberry-pi.md) · [Docs index](README.md)

The Raspberry Pi Zero 2 W (quad Cortex-A53 at 1 GHz, 512 MB RAM, about $15) is the cheapest target hae is designed for. All numbers below were **measured on the device**: Zero 2 W Rev 1.0, Raspberry Pi OS Bookworm 64-bit, Python 3.11.2, `performance` governor, no heatsink, onnxruntime CPU. Times are the median (p50) over repeated runs.

## Install

```bash
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/hae-env && . ~/hae-env/bin/activate
pip install numpy onnxruntime opencv-python-headless pillow onnx
pip install --no-deps "hae @ git+https://github.com/sashapiscivca-svg/hae"      # runtime only, no PyTorch
# copy models/hae-s-ir/hae-s-ir_lwir_int8.onnx (383 KB) to the Pi
```

A 64-bit OS is required, because onnxruntime publishes aarch64 wheels. Peak memory of a full detector-plus-tracker pipeline was 175–185 MB of the 416 MB available to the OS.

## Measured latency

| hae-s-ir | 1 thread | 4 threads |
|---|---:|---:|
| int8, one 256×320 tile | 148 ms | 55 ms |
| fp32, one 256×320 tile | 208 ms | 82 ms |
| int8, full 640×512 frame at ×1.5 | 1411 ms | 554 ms |

Use the int8 model: it is about 30 % faster than fp32 on this CPU, with the same accuracy. Decoding takes about 7.5 ms per tile in numpy. On this board, latency was roughly 16–40× higher than on one x86 laptop thread, so do not extrapolate from desktop timings.

Reproduce the measurements with:

```bash
python scripts/bench.py models/hae-s-ir/hae-s-ir_lwir_int8.onnx --threads 1
python scripts/bench.py models/hae-s-ir/hae-s-ir_lwir_int8.onnx --threads 4
```

## Thermal and power

Without a heatsink, all four cores under continuous load reached **81.7 °C** within 5 minutes, and the clock throttled to 834 MHz. With 1–3 busy cores the board stayed around 56 °C. For flight:

- fit a heatsink (and some airflow if possible);
- limit inference to 3 threads (`Detector(..., threads=3)`) and leave a core for capture and telemetry;
- watch `vcgencmd measure_temp` and `vcgencmd get_throttled`.

We did not measure power draw.

## Real-time strategy

A thermal camera delivers 9–30 frames per second, while hae-s-ir needs about 0.55 s for a full 640×512 frame at ×1.5 on four threads. Do not run the detector on every frame:

1. Run the detector every *K* frames (K = 3–8), and choose *K* adaptively from a time budget of, say, 150–200 ms per frame.
2. Between detections, follow each candidate with a cheap tracker (a correlation filter or optical flow).
3. Alert the operator only when a candidate persists across several detections. This also suppresses most single-frame false alarms.

In our pipeline tests on the device, this scheme ran at about 7 frames per second with an `s`-size thermal detector (and 6–12 frames per second with a smaller one, depending on *K*).

Other ways to save time:

- **Lower the scale** (`scale=1.0` or `1.25`) when flying lower, or when people appear larger than about 25 px. The cost falls with the square of the scale.
- **Region of interest:** run on a crop of the frame (for example below the horizon).

## Minimal on-board loop

```python
import cv2
from hae.detector import Detector

det = Detector("models/hae-s-ir/hae-s-ir_lwir_int8.onnx", threads=3)   # recommended scale and threshold from metadata
cap = cv2.VideoCapture(0)                                               # a thermal camera as a V4L2 device
k, K = 0, 5
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if k % K == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        dets = det.detect(gray)                    # (n, 6): x1, y1, x2, y2, score, class
        for x1, y1, x2, y2, s, _ in dets:
            print(f"candidate {s:.2f} at {(x1 + x2) / 2:.0f},{(y1 + y2) / 2:.0f}")
    k += 1
```

Thermal input must be the camera's **8-bit grayscale** image (a white-hot palette). Radiometric 14/16-bit data needs an AGC-like conversion to 8-bit first. Colour palettes (ironbow and similar) are not supported.
