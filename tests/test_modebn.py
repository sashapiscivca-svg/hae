
import torch

from hae.export import export_onnx
from hae.model.presets import build


def _inputs():
    g = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 64, 64, generator=g), torch.rand(2, 1, 64, 64, generator=g)


def _perturb(m):   # random weights (non-zero head signal) and different BN statistics per mode
    g = torch.Generator().manual_seed(1)
    for p in m.parameters():
        p.data = torch.randn(p.shape, generator=g) * 0.3
    for mod in m.modules():
        if hasattr(mod, "bns"):
            for k, bn in enumerate(mod.bns):
                bn.running_mean.fill_(0.1 * k)
                bn.running_var.fill_(1.0 + k)


def test_modebn_modes_use_separate_statistics():
    m = build("n", overrides={"modebn": True}).eval()
    _perturb(m)
    rgb, t = _inputs()
    with torch.no_grad():
        before = {k: m(**kw)["box"].clone() for k, kw in (("both", dict(rgb=rgb, t=t)), ("rgb", dict(rgb=rgb)), ("lwir", dict(t=t)))}
        for mod in m.modules():
            if hasattr(mod, "bns"):
                mod.bns[1].running_mean.add_(5.0)     # touch only the rgb-mode statistics
        after = {k: m(**kw)["box"] for k, kw in (("both", dict(rgb=rgb, t=t)), ("rgb", dict(rgb=rgb)), ("lwir", dict(t=t)))}
    assert not torch.equal(before["rgb"], after["rgb"])
    assert torch.equal(before["lwir"], after["lwir"]) and torch.equal(before["both"], after["both"])


def test_for_mode_collapses_to_plain_bn_with_identical_output():
    m = build("n", overrides={"modebn": True}).eval()
    _perturb(m)
    rgb, t = _inputs()
    for mode, kw in (("both", dict(rgb=rgb, t=t)), ("rgb", dict(rgb=rgb)), ("lwir", dict(t=t))):
        c = m.for_mode(mode)
        assert not any(hasattr(x, "bns") for x in c.modules())
        with torch.no_grad():
            assert torch.allclose(c(**kw)["box"], m(**kw)["box"], atol=1e-6)


def test_modebn_onnx_export_per_mode(tmp_path):
    m = build("n", block="repghost", overrides={"modebn": True}).eval()
    _perturb(m)
    r = export_onnx(m, tmp_path / "rgb.onnx", mode="rgb", hw=(64, 64))
    assert r["bytes"] > 0
