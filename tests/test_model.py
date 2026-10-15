import pytest
import torch

from hae.model.hae import HaeConfig, HaeNet


def _inputs(b=2, h=256, w=320):
    return torch.rand(b, 3, h, w), torch.rand(b, 1, h, w)


@pytest.mark.parametrize("block", ["dense", "dw", "ghost", "rep", "repghost"])
def test_forward_shapes_all_modes(block):
    m = HaeNet(HaeConfig(block=block))
    rgb, t = _inputs()
    for kw in ({"rgb": rgb, "t": t}, {"rgb": rgb}, {"t": t}):
        out = m(**kw)
        assert out["cls"].shape == (2, 1, 64, 80)
        assert out["box"].shape == (2, 4, 64, 80)
        assert torch.isfinite(out["cls"]).all() and torch.isfinite(out["box"]).all()


def test_single_modality_gradients_reach_only_its_stem():
    m = HaeNet(HaeConfig())
    rgb, t = _inputs()
    out = m(rgb=rgb)
    (out["cls"].sum() + out["box"].sum()).backward()
    assert m.stem_rgb.net[0].weight.grad is not None
    assert m.stem_t.net[0].weight.grad is None
    m.zero_grad(set_to_none=True)
    m(t=t)["cls"].sum().backward()
    assert m.stem_t.net[0].weight.grad is not None and m.stem_rgb.net[0].weight.grad is None


def test_both_modalities_use_gate_and_both_stems():
    m = HaeNet(HaeConfig())
    rgb, t = _inputs()
    m(rgb=rgb, t=t)["cls"].sum().backward()
    assert m.stem_rgb.net[0].weight.grad is not None and m.stem_t.net[0].weight.grad is not None
    assert m.gate.conv.weight.grad is not None


def test_size_not_multiple_of_16_raises():
    m = HaeNet(HaeConfig())
    with pytest.raises(ValueError, match="16"):
        m(rgb=torch.rand(1, 3, 250, 330))


def test_no_input_raises():
    with pytest.raises(ValueError):
        HaeNet(HaeConfig())()


def test_zero_input_is_finite_and_cls_prior_is_low():
    m = HaeNet(HaeConfig()).eval()
    with torch.no_grad():
        out = m(rgb=torch.zeros(1, 3, 64, 64), t=torch.zeros(1, 1, 64, 64))
    assert torch.isfinite(out["cls"]).all()
    assert float(out["cls"].sigmoid().mean()) < 0.05  # initialization: low prior probability


def test_fuse_keeps_outputs_for_rep_block():
    torch.manual_seed(0)
    m = HaeNet(HaeConfig(block="rep"))
    rgb, t = _inputs(2, 64, 64)
    for _ in range(3):
        m(rgb=rgb, t=t)
    m.eval()
    a = m(rgb=rgb, t=t)
    m.fuse()
    b = m(rgb=rgb, t=t)
    assert torch.allclose(a["cls"], b["cls"], atol=1e-3) and torch.allclose(a["box"], b["box"], atol=1e-3)


def test_param_count_is_tiny():
    n = sum(p.numel() for p in HaeNet(HaeConfig()).parameters())
    assert n < 400_000, n


def test_scaled_identity_and_growth():
    base = HaeConfig()
    assert base.scaled(1.0, 1.0) == base
    big = base.scaled(2.0, 2.0)
    assert all(c % 8 == 0 for c in (big.stem_c, big.neck_c, big.head_c, *big.widths))
    assert big.depths == (2, 4, 4)

    def n(c):
        return sum(p.numel() for p in HaeNet(c).parameters())

    assert n(big) > 3 * n(base)


def test_scaled_model_runs_all_blocks():
    for kind in ("ghost", "dense"):
        m = HaeNet(HaeConfig(block=kind).scaled(1.5, 1.5)).eval()
        with torch.no_grad():
            out = m(rgb=torch.rand(1, 3, 64, 64), t=torch.rand(1, 1, 64, 64))
        assert out["cls"].shape == (1, 1, 16, 16)


def test_multiclass_head_shapes_and_scaled_keeps_ncls():
    m = HaeNet(HaeConfig(ncls=3))
    out = m(rgb=torch.rand(1, 3, 64, 64))
    assert out["cls"].shape == (1, 3, 16, 16) and out["box"].shape == (1, 4, 16, 16)
    assert HaeConfig(ncls=3).scaled(2.0, 1.0).ncls == 3
    assert float(m.cls.bias.sigmoid().mean()) < 0.02


def test_ctx32_stage_shapes_cost_and_size_rule():
    from torch.utils.flop_counter import FlopCounterMode

    def gmac(cfg):
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            HaeNet(cfg).eval()(rgb=torch.rand(1, 3, 256, 320), t=torch.rand(1, 1, 256, 320))
        return fc.get_total_flops() / 2e9

    base, with_ctx = gmac(HaeConfig()), gmac(HaeConfig(ctx=True))
    assert base < with_ctx < 1.08 * base, (base, with_ctx)       # the context level is almost free
    m = HaeNet(HaeConfig(ctx=True, ncls=3))
    out = m(rgb=torch.rand(1, 3, 64, 96), t=torch.rand(1, 1, 64, 96))
    assert out["cls"].shape == (1, 3, 16, 24)
    (out["cls"].sum() + out["box"].sum()).backward()
    assert m.s32[0][0][0].weight.grad is not None
    with pytest.raises(ValueError, match="32"):
        m(rgb=torch.rand(1, 3, 48, 80))                         # multiple of 16 but not of 32
    assert HaeConfig(ctx=True).scaled(2.0, 1.0).ctx is True


def test_aux_head_exists_only_in_training_and_not_in_onnx_path():
    m = HaeNet(HaeConfig(aux=True, ncls=3))
    x = dict(rgb=torch.rand(1, 3, 64, 96), t=torch.rand(1, 1, 64, 96))
    m.train()
    out = m(**x)
    assert out["aux"].shape == (1, 1, 16, 24)
    m.eval()
    assert "aux" not in m(**x)                               # no auxiliary branch at runtime
    assert HaeConfig(aux=True).scaled(2.0, 1.0).aux is True and "aux" not in HaeNet(HaeConfig()).train()(**x)
