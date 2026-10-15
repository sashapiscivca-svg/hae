import numpy as np
import pytest
import torch

from hae.export import export_onnx
from hae.model.presets import build

ort = pytest.importorskip("onnxruntime")


@pytest.mark.parametrize("mode", ["both", "rgb", "lwir"])
def test_onnx_parity_for_each_mode(tmp_path, mode):
    torch.manual_seed(0)
    m = build(size="n", ncls=3).eval()
    p = tmp_path / f"n_{mode}.onnx"
    export_onnx(m, p, mode=mode, hw=(64, 96))
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    rgb, t = torch.rand(1, 3, 64, 96), torch.rand(1, 1, 64, 96)
    feeds = {"rgb": rgb.numpy(), "t": t.numpy()}
    feeds = {k: v for k, v in feeds.items() if k in [i.name for i in sess.get_inputs()]}
    out = sess.run(None, feeds)
    with torch.no_grad():
        ref = m(**{k: {"rgb": rgb, "t": t}[k] for k in feeds})
    assert np.allclose(out[0], ref["cls"].numpy(), atol=1e-4) and np.allclose(out[1], ref["box"].numpy(), atol=1e-4)
    assert set(i.name for i in sess.get_inputs()) == ({"rgb", "t"} if mode == "both" else {"rgb"} if mode == "rgb" else {"t"})


def test_onnx_accepts_other_tile_sizes_and_reports_size(tmp_path):
    m = build(size="n", ncls=3).eval()
    p = tmp_path / "n.onnx"
    info = export_onnx(m, p, mode="lwir", hw=(64, 96))
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"t": np.random.rand(1, 1, 128, 160).astype(np.float32)})
    assert out[0].shape == (1, 3, 32, 40) and out[1].shape == (1, 4, 32, 40)
    assert info["bytes"] == p.stat().st_size and info["params"] > 0


def test_repghost_model_exports_fused_with_onnx_parity(tmp_path):
    torch.manual_seed(0)
    m = build(size=None, block="repghost", width=0.67, depth=1.0, ncls=3)
    for _ in range(4):                                   # BN statistics in multi-branch blocks
        m(rgb=torch.rand(4, 3, 64, 96), t=torch.rand(4, 1, 64, 96))
    m.eval()
    p = tmp_path / "rg.onnx"
    export_onnx(m, p, mode="both", hw=(64, 96))
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    rgb, t = torch.rand(1, 3, 64, 96), torch.rand(1, 1, 64, 96)
    out = sess.run(None, {"rgb": rgb.numpy(), "t": t.numpy()})
    with torch.no_grad():
        ref = m(rgb=rgb, t=t)
    assert np.allclose(out[0], ref["cls"].numpy(), atol=1e-4) and np.allclose(out[1], ref["box"].numpy(), atol=1e-4)
    # no training-time branches in the graph: ONNX size = plain ghost of the same width (+- bias)
    g = tmp_path / "g.onnx"
    export_onnx(build(size=None, block="ghost", width=0.67, depth=1.0, ncls=3).eval(), g, mode="both", hw=(64, 96))
    assert abs(p.stat().st_size - g.stat().st_size) < 0.1 * g.stat().st_size
