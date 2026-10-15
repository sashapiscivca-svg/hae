import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")
from hae.export import export_onnx          # noqa: E402
from hae.model.presets import build         # noqa: E402
from hae.quantize import quantize_int8      # noqa: E402


def test_int8_is_smaller_and_close_to_fp32(tmp_path):
    torch.manual_seed(0)
    m = build(size="s", ncls=3).eval()                 # s: enough weights for the size gain to show
    fp = tmp_path / "s.onnx"
    export_onnx(m, fp, mode="lwir", hw=(64, 96))
    calib = [np.random.default_rng(i).random((1, 1, 64, 96), dtype=np.float32) for i in range(8)]
    q = tmp_path / "s_int8.onnx"
    info = quantize_int8(fp, q, {"t": calib})
    assert q.stat().st_size < 0.6 * fp.stat().st_size and info["bytes_int8"] == q.stat().st_size
    x = np.random.default_rng(99).random((1, 1, 64, 96), dtype=np.float32)
    a = ort.InferenceSession(str(fp), providers=["CPUExecutionProvider"]).run(None, {"t": x})
    b = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"]).run(None, {"t": x})
    for u, v in zip(a, b):
        assert u.shape == v.shape and float(np.abs(u - v).max()) < 0.05   # untrained network: cls is nearly constant, so compare absolutely


def test_missing_calibration_input_raises(tmp_path):
    m = build(size="n", ncls=3).eval()
    fp = tmp_path / "n.onnx"
    export_onnx(m, fp, mode="both", hw=(64, 96))
    with pytest.raises(ValueError, match="rgb"):
        quantize_int8(fp, tmp_path / "q.onnx", {"t": [np.zeros((1, 1, 64, 96), np.float32)]})
