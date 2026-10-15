import torch

from hae.model.align import Align, edge_alignment_loss, edge_map, flow_smoothness, warp
from hae.model.hae import HaeConfig, HaeNet


def test_warp_shifts_content_by_flow_cells():
    f = torch.zeros(1, 1, 8, 10)
    f[0, 0, 5, 7] = 1.0
    flow = torch.zeros(1, 2, 8, 10)
    flow[:, 0] = 2.0          # dx = +2 cells: out(x) = f(x + 2)
    out = warp(f, flow)
    assert out[0, 0, 5, 5] > 0.99 and out[0, 0, 5, 7] < 0.01


def test_align_is_identity_at_init_and_keeps_shape():
    torch.manual_seed(0)
    a = Align(16, stride=4, max_shift_px=48)
    fr, ft = torch.rand(2, 16, 16, 20), torch.rand(2, 16, 16, 20)
    out, flow = a(fr, ft)
    assert out.shape == fr.shape and flow.shape == (2, 2, 16, 20)
    assert torch.allclose(out, fr, atol=1e-6) and flow.abs().max() < 1e-6


def test_edge_loss_prefers_true_shift():
    torch.manual_seed(1)
    base = torch.nn.functional.avg_pool2d(torch.rand(2, 1, 72, 104), 3, 1, 1)
    rgb = base.repeat(1, 3, 1, 1)
    t = torch.roll(base, shifts=(-4, -8), dims=(2, 3))      # t(x) = rgb(x + d), d = (dx=8, dy=4) px
    true = torch.zeros(2, 2, 18, 26); true[:, 0] = 8.0; true[:, 1] = 4.0
    zero, wrong = torch.zeros_like(true), -true
    l_true, l_zero, l_wrong = (edge_alignment_loss(rgb, t, f, 4) for f in (true, zero, wrong))
    # frame borders leave a residual (roll wraps, warp repeats the edge), so check ordering, not zero
    assert l_true < l_zero - 0.03 and l_true < l_wrong - 0.03, (float(l_true), float(l_zero), float(l_wrong))


def test_flow_smoothness():
    const = torch.ones(1, 2, 8, 8)
    noisy = torch.rand(1, 2, 8, 8)
    assert flow_smoothness(const) == 0.0 and flow_smoothness(noisy) > 0.05


def test_edge_map_shape_and_normalised():
    e = edge_map(torch.rand(2, 3, 64, 96), 4)
    assert e.shape == (2, 1, 16, 24) and abs(float(e.mean())) < 1e-4


def test_haenet_with_align_outputs_flow_only_for_both_and_trains():
    cfg = HaeConfig(align=True)
    m = HaeNet(cfg)
    rgb, t = torch.rand(2, 3, 64, 96), torch.rand(2, 1, 64, 96)
    out = m(rgb=rgb, t=t)
    assert out["flow"].shape == (2, 2, 16, 24)
    assert "flow" not in m(rgb=rgb) and "flow" not in m(t=t)
    (out["cls"].sum() + edge_alignment_loss(rgb, t, out["flow"], 4)).backward()
    assert m.align.net[0].weight.grad is not None
    base = sum(p.numel() for p in HaeNet(HaeConfig()).parameters())
    assert 0 < sum(p.numel() for p in m.parameters()) - base < 0.3 * base
    assert HaeConfig(align=True).scaled(2.0, 1.0).align is True
