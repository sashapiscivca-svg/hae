import numpy as np

from hae.data.sampler import fourier_low_amp, fourier_mix


def _img(seed, mean):
    r = np.random.default_rng(seed)
    return np.clip(r.normal(mean, 25, (64, 80, 3)), 0, 255).astype(np.uint8)


def test_lam_zero_is_identity_and_output_is_valid():
    x = _img(0, 100)
    bank = fourier_low_amp(_img(1, 200), band=0.1)
    assert np.array_equal(fourier_mix(x, bank, lam=0.0, band=0.1), x)
    y = fourier_mix(x, bank, lam=1.0, band=0.1)
    assert y.dtype == np.uint8 and y.shape == x.shape


def test_low_frequency_mix_moves_global_tone_toward_partner_but_keeps_detail():
    x, partner = _img(0, 70), _img(1, 190)
    bank = fourier_low_amp(partner, band=0.1)
    y = fourier_mix(x, bank, lam=1.0, band=0.1)
    assert abs(y.mean() - partner.mean()) < abs(x.mean() - partner.mean()) - 30       # overall tone moved towards the partner
    assert np.corrcoef(x.astype(float).ravel(), y.astype(float).ravel())[0, 1] > 0.6  # fine details (phase) preserved


def test_bank_is_small_low_frequency_block():
    bank = fourier_low_amp(_img(2, 120), band=0.1)
    assert bank.shape[0] == 3 and bank.shape[1] < 64 and bank.shape[2] < 80


def test_train_runs_with_fourier_and_photo(tmp_path):
    import json
    from PIL import Image
    from hae.data.cache import build_cache
    from hae.train import TrainConfig, train
    root = tmp_path / "ds"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(6):
        im = np.random.default_rng(i).integers(0, 255, (96, 128, 3), np.uint8)
        im[30:46, 20 + 12 * i:36 + 12 * i] = 255
        for m in ("visible", "thermal"):
            Image.fromarray(im).save(root / "train" / m / f"{i}.jpg", quality=100)
        imgs.append({"id": i, "file_name": f"{i}.jpg", "height": 96, "width": 128})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [20 + 12 * i, 30, 16, 16], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", tmp_path / "c")
    cfg = TrainConfig(cache_dir=str(tmp_path / "c"), ann_root=str(root), out_dir=str(tmp_path / "r"), epochs=1, crops_per_epoch=16, batch=8,
                      threads=2, crop_hw=(64, 64), size="n", classes=1, prefetch=0, mode_probs=(0, 1, 0), fourier_p=0.7, photo=0.7)
    assert train(cfg)["history"][-1]["loss"] > 0


def test_fast_version_matches_full_fft_closely_and_is_much_faster():
    import time
    from hae.data.sampler import fourier_mix_fast
    x, partner = _img(0, 70), _img(1, 190)
    big = np.kron(x, np.ones((4, 4, 1), np.uint8))[:256, :320]       # 256x320, smooth upscaling
    pbig = np.kron(partner, np.ones((4, 4, 1), np.uint8))[:256, :320]
    bank = fourier_low_amp(pbig, band=0.1)
    full = fourier_mix(big, bank, 0.8, 0.1)
    fast = fourier_mix_fast(big, bank, 0.8, 0.1)
    assert abs(float(full.mean()) - float(fast.mean())) < 6                       # same tone shift
    assert np.abs(full.astype(float) - fast.astype(float)).mean() < 8             # similar images
    t0 = time.perf_counter(); [fourier_mix(big, bank, 0.8, 0.1) for _ in range(20)]; t_full = time.perf_counter() - t0
    t0 = time.perf_counter(); [fourier_mix_fast(big, bank, 0.8, 0.1) for _ in range(20)]; t_fast = time.perf_counter() - t0
    assert t_fast < t_full / 3
