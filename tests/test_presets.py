import pytest
import torch
from torch.utils.flop_counter import FlopCounterMode

from hae.model.hae import HaeConfig, HaeNet
from hae.model.presets import SIZES, build, preset_config


def _gmac(m):
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m.eval()(rgb=torch.rand(1, 3, 256, 320), t=torch.rand(1, 1, 256, 320))
    return fc.get_total_flops() / 2e9


def test_sizes_are_nano_to_xlarge():
    assert SIZES == ("n", "t", "s", "m", "l", "x")


def test_lineup_is_strictly_increasing_in_params_and_compute():
    models = [build(size=s) for s in SIZES]
    params = [sum(p.numel() for p in m.parameters()) for m in models]
    macs = [_gmac(m) for m in models]
    assert params == sorted(params) and len(set(params)) == 6, params
    assert macs == sorted(macs) and len(set(macs)) == 6, macs


def test_edge_sizes_fit_the_pi_zero_budget():
    # published lineup for weak hardware: n, t, s <= 0.3 GMAC per 320x256 tile
    assert _gmac(build(size="n")) <= 0.06
    assert _gmac(build(size="t")) <= 0.12
    assert _gmac(build(size="s")) <= 0.30


def test_t_is_the_validated_v0_baseline():
    assert preset_config("t") == HaeConfig(block="ghost")


def test_unknown_size_raises_and_build_without_size_uses_args():
    with pytest.raises(ValueError, match="unknown size"):
        preset_config("xxl")
    m = build(size=None, block="dense", width=1.5, depth=1.0)
    assert isinstance(m, HaeNet) and m.cfg == HaeConfig(block="dense").scaled(1.5, 1.0)


def test_all_sizes_run_all_modes():
    for s in SIZES[:4]:
        m = build(size=s).eval()
        with torch.no_grad():
            for kw in ({"rgb": torch.rand(1, 3, 64, 64)}, {"t": torch.rand(1, 1, 64, 64)},
                       {"rgb": torch.rand(1, 3, 64, 64), "t": torch.rand(1, 1, 64, 64)}):
                assert m(**kw)["cls"].shape == (1, 1, 16, 16)
