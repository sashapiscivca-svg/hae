import torch

from hae.export import export_onnx
from hae.loss import HaeLoss
from hae.model.presets import build


def test_fine_stem_keeps_stride4_output_and_adds_macs():
    base, fine = build("n"), build("n", overrides={"fine_stem": True})
    x = torch.rand(1, 1, 64, 96)
    assert fine(t=x)["cls"].shape == base(t=x)["cls"].shape == (1, 1, 16, 24)
    assert sum(p.numel() for p in fine.parameters()) > sum(p.numel() for p in base.parameters())


def test_head_s2_outputs_stride2_grid_and_config_follows():
    m = build("n", overrides={"head_s2": True})
    assert m.cfg.stride == 2 and m.cfg.fine_stem
    out = m(t=torch.rand(2, 1, 64, 96))
    assert out["cls"].shape[-2:] == (32, 48) and out["box"].shape[-2:] == (32, 48)


def test_head_s2_rejects_fusion_mode_and_loss_uses_stride():
    import pytest
    m = build("n", overrides={"head_s2": True})
    with pytest.raises(ValueError):
        m(rgb=torch.rand(1, 3, 64, 96), t=torch.rand(1, 1, 64, 96))
    out = m(t=torch.rand(1, 1, 64, 96))
    loss, _ = HaeLoss(stride=m.cfg.stride)(out, [torch.tensor([[10.0, 10.0, 18.0, 19.0]])], [torch.zeros(0, 4)])
    assert torch.isfinite(loss)


def test_head_s2_exports_to_onnx(tmp_path):
    m = build("n", block="repghost", overrides={"head_s2": True}).eval()
    assert export_onnx(m, tmp_path / "s2.onnx", mode="lwir", hw=(64, 96))["bytes"] > 0
