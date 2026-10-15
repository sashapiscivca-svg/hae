import torch
from torch.utils.flop_counter import FlopCounterMode

from hae.model.blob import BlobBank
from hae.model.hae import HaeConfig, HaeNet


def _disc(h=64, w=64, cy=32, cx=32, r=4):
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return (((ys - cy) ** 2 + (xs - cx) ** 2) <= r * r).float()[None, None]


def test_output_shape_and_zero_response_on_constant_image():
    bank = BlobBank(scales=(1.5, 2.5))
    out = bank(torch.full((2, 3, 48, 64), 0.7))
    assert out.shape == (2, 2, 48, 64)
    assert out[:, :, 8:-8, 8:-8].abs().max() < 1e-4        # DoG of a constant image = 0 (away from the border)


def test_blob_scale_selectivity():
    bank = BlobBank(scales=(1.5, 2.5, 4.0))
    best = []
    for r in (2, 4, 6):    # blob radius -> index of the scale with the strongest response at the centre
        out = bank(_disc(r=r).repeat(1, 3, 1, 1))
        best.append(int(out[0, :, 32, 32].argmax()))
    assert best == [0, 1, 2], best


def test_bank_has_no_trainable_parameters_and_haenet_accepts_blob_flag():
    assert sum(p.numel() for p in BlobBank().parameters()) == 0
    m = HaeNet(HaeConfig(blob=True))
    rgb, t = torch.rand(1, 3, 64, 96), torch.rand(1, 1, 64, 96)
    for kw in ({"rgb": rgb, "t": t}, {"rgb": rgb}, {"t": t}):
        assert m(**kw)["cls"].shape == (1, 1, 16, 24)
    assert HaeConfig(blob=True).scaled(2.0, 1.0).blob is True


def test_blob_overhead_is_moderate_for_hae_t():
    def gmac(cfg):
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            HaeNet(cfg).eval()(rgb=torch.rand(1, 3, 256, 320), t=torch.rand(1, 1, 256, 320))
        return fc.get_total_flops() / 2e9
    base, with_blob = gmac(HaeConfig()), gmac(HaeConfig(blob=True))
    assert base < with_blob < 1.5 * base, (base, with_blob)
