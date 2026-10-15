"""Latency of hae ONNX models per tile in onnxruntime, 1 thread by default (the Raspberry Pi Zero 2 W setting).
Needs only numpy and onnxruntime, so it runs on the Pi as is (decoding is timed when the hae package is importable).

    python scripts/bench.py models/hae-s-ir/hae-s-ir_lwir.onnx models/hae-s-ir/hae-s-ir_lwir_int8.onnx
    python scripts/bench.py model.onnx --tile 512x640 --n 50 --threads 4
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "hae").is_dir():
    sys.path.insert(0, str(ROOT / "src"))


def bench(path: str, hw=(256, 320), n: int = 30, threads: int = 1) -> dict:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feeds = {i.name: np.random.rand(1, 3 if i.name == "rgb" else 1, *hw).astype(np.float32) for i in s.get_inputs()}
    for _ in range(3):
        s.run(None, feeds)
    t = []
    for _ in range(n):
        a = time.perf_counter()
        cls, box = s.run(None, feeds)
        t.append(time.perf_counter() - a)
    rep = {"file": str(path), "size_kb": round(os.path.getsize(path) / 1024), "ms_p50": round(1000 * float(np.median(t)), 1),
           "ms_p90": round(1000 * float(np.percentile(t, 90)), 1)}
    try:
        from hae.decode_np import decode_np
        a = time.perf_counter()
        decode_np(cls, box)
        rep["decode_ms"] = round(1000 * (time.perf_counter() - a), 2)
    except ImportError:
        pass
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-tile latency of hae ONNX models (onnxruntime, CPU).")
    ap.add_argument("models", nargs="+", help="ONNX files")
    ap.add_argument("--tile", default="256x320", help="input HxW")
    ap.add_argument("--n", type=int, default=30, help="timed runs")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    hw = tuple(int(v) for v in a.tile.lower().split("x"))
    print(f"CPU: {platform.processor() or platform.machine()} | onnxruntime {ort.__version__} | {a.threads} thread(s) | tile {hw[0]}x{hw[1]}")
    for p in a.models:
        print(bench(p, hw, a.n, a.threads), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
