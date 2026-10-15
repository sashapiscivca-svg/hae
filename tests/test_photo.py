import numpy as np

from hae.data.sampler import photometric


def _img(v=120):
    return np.full((32, 48, 3), v, np.uint8)


def test_zero_strength_is_identity_and_output_is_valid_uint8():
    rng = np.random.default_rng(0)
    x = np.random.default_rng(1).integers(0, 255, (32, 48, 3), np.uint8)
    assert np.array_equal(photometric(x, rng, 0.0), x)
    y = photometric(x, rng, 1.0)
    assert y.dtype == np.uint8 and y.shape == x.shape


def test_strong_photometric_changes_brightness_distribution_but_stays_bounded():
    rng = np.random.default_rng(0)
    means = np.array([photometric(_img(), rng, 1.0).mean() for _ in range(200)])
    assert means.std() > 8 and 0 <= means.min() and means.max() <= 255   # real brightness spread across crops


def test_grayscale_option_equalizes_channels():
    rng = np.random.default_rng(3)
    x = np.dstack([np.full((16, 16), 200, np.uint8), np.full((16, 16), 50, np.uint8), np.full((16, 16), 90, np.uint8)])
    outs = [photometric(x, rng, 1.0) for _ in range(300)]
    gray = [o for o in outs if np.abs(o[..., 0].astype(int) - o[..., 1]).max() <= 2 and np.abs(o[..., 1].astype(int) - o[..., 2]).max() <= 2]
    assert 5 < len(gray) < 150   # gray crops occur, but not always


def test_color_strength_zero_never_goes_grayscale_but_still_varies_tone():
    rng = np.random.default_rng(0)
    x = np.dstack([np.full((16, 16), 200, np.uint8), np.full((16, 16), 60, np.uint8), np.full((16, 16), 90, np.uint8)])
    outs0 = [photometric(x, rng, 1.0, color=0.0) for _ in range(300)]
    outs1 = [photometric(x, rng, 1.0, color=1.0) for _ in range(300)]
    gray = lambda o: np.abs(o[..., 0].astype(int) - o[..., 1]).max() <= 2 and np.abs(o[..., 1].astype(int) - o[..., 2]).max() <= 2
    assert sum(gray(o) for o in outs0) == 0 and sum(gray(o) for o in outs1) > 5     # grayscale only with the colour part
    assert np.array([o.mean() for o in outs0]).std() > 6                              # tone still varies
    assert all(o[..., 0].mean() > o[..., 2].mean() > o[..., 1].mean() for o in outs0)  # channel order (hue) preserved
