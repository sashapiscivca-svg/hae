import pytest
import torch

from hae.model.blocks import Down, fuse_model, make_block


@pytest.mark.parametrize("kind", ["dense", "dw", "ghost", "rep"])
def test_block_keeps_shape_and_has_grad(kind):
    b = make_block(kind, 24)
    x = torch.randn(2, 24, 16, 20, requires_grad=True)
    y = b(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_down_halves_resolution():
    assert Down(24, 32)(torch.randn(1, 24, 16, 20)).shape == (1, 32, 8, 10)


def test_repblock_fuse_is_equivalent():
    torch.manual_seed(0)
    b = make_block("rep", 16)
    for _ in range(5):  # give BatchNorm non-trivial statistics
        b(torch.randn(8, 16, 12, 12))
    b.eval()
    x = torch.randn(2, 16, 12, 12)
    before = b(x)
    fuse_model(b)
    after = b(x)
    assert torch.allclose(before, after, atol=1e-4), float((before - after).abs().max())


def test_unknown_kind_raises():
    with pytest.raises(KeyError):
        make_block("nope", 8)


def test_repghost_fuse_is_equivalent_and_costs_the_same_as_ghost_after_fuse():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    from hae.model.blocks import fuse_model, make_block
    torch.manual_seed(0)
    b = make_block("repghost", 16)
    for _ in range(6):
        b(torch.randn(8, 16, 12, 12))        # BN statistics
    b.eval()
    x = torch.randn(2, 16, 12, 12)
    before = b(x)
    fuse_model(b)
    assert torch.allclose(before, b(x), atol=1e-4), float((before - b(x)).abs().max())
    g = make_block("ghost", 16).eval()
    with FlopCounterMode(display=False) as f1, torch.no_grad():
        b(x)
    with FlopCounterMode(display=False) as f2, torch.no_grad():
        g(x)
    assert f1.get_total_flops() == f2.get_total_flops()
    y = make_block("repghost", 16)(x)
    y.sum().backward()
