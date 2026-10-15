import torch

from hae.export import export_onnx
from hae.model.hae import local_norm
from hae.model.presets import build


def test_local_norm_removes_slow_background_and_keeps_small_blobs():
    h, w = 64, 96
    ramp = torch.linspace(0, 8, w).view(1, 1, 1, w).expand(1, 1, h, w).clone()   # slow background gradient
    x = ramp.clone()
    x[..., 30:34, 40:44] += 2.0                                                   # a tiny "warm spot"
    y = local_norm(x)
    assert y.shape == x.shape
    far = y[..., 8:16, 8:16].abs().mean()                                          # background without spots -> ~0
    blob = y[..., 30:34, 40:44].mean()
    assert far < 0.2 and blob > 1.0 and blob > 5 * far


def test_lnorm_model_has_extra_thermal_channel_and_exports(tmp_path):
    m = build("n", overrides={"lnorm": True}).eval()
    assert sum(p.numel() for p in m.parameters()) > sum(p.numel() for p in build("n").parameters())   # extra input channel in stem_t
    out = m(t=torch.rand(1, 1, 64, 96))
    assert out["cls"].shape[-2:] == (16, 24)
    r = export_onnx(m, tmp_path / "ln.onnx", mode="lwir", hw=(64, 96))
    assert r["bytes"] > 0
