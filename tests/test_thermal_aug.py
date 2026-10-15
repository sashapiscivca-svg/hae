import numpy as np

from hae.data.sampler import thermal_aug


def _t(seed=0):
    r = np.random.default_rng(seed)
    t = np.clip(r.normal(90, 12, (64, 80)), 0, 255).astype(np.uint8)
    t[20:28, 30:38] = 220
    return t


def test_zero_strength_is_identity_and_output_is_valid():
    t = _t()
    rng = np.random.default_rng(0)
    assert np.array_equal(thermal_aug(t, rng, 0.0), t)
    y = thermal_aug(t, rng, 1.0)
    assert y.dtype == np.uint8 and y.shape == t.shape


def test_changes_shape_of_histogram_but_keeps_global_gain_and_offset():
    t = _t()
    rng = np.random.default_rng(1)
    outs = [thermal_aug(t, rng, 1.0) for _ in range(80)]
    assert np.mean([abs(float(o.mean()) - float(t.mean())) for o in outs]) < 3.0         # global shift removed (frame normalization removes it anyway)
    assert np.mean([abs(float(o.std()) - float(t.std())) for o in outs]) < 3.5
    diffs = np.array([np.abs(o.astype(float) - t).mean() for o in outs])
    assert diffs.mean() > 1.5 and diffs.std() > 0.3                                       # augmentation really changes the frame, and differently each time
    assert min(np.corrcoef(o.astype(float).ravel(), t.astype(float).ravel())[0, 1] for o in outs) > 0.6    # the person stays recognizable


def test_train_runs_with_thermal_aug(tmp_path):
    import json

    from PIL import Image

    from hae.data.cache import build_cache
    from hae.train import TrainConfig, train
    root = tmp_path / "ds"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(4):
        im = np.random.default_rng(i).integers(0, 120, (64, 96, 3), np.uint8)
        im[20:30, 10 + 12 * i:20 + 12 * i] = 255
        for m in ("visible", "thermal"):
            Image.fromarray(im).save(root / "train" / m / f"{i}.jpg", quality=100)
        imgs.append({"id": i, "file_name": f"{i}.jpg", "height": 64, "width": 96})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [10 + 12 * i, 20, 10, 10], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", tmp_path / "c")
    cfg = TrainConfig(cache_dir=str(tmp_path / "c"), ann_root=str(root), out_dir=str(tmp_path / "r"), epochs=1, crops_per_epoch=16, batch=8, threads=2,
                      crop_hw=(32, 64), size="n", classes=1, prefetch=0, mode_probs=(0, 0, 1), tnorm=True, taug=0.8, tb=False)
    assert train(cfg)["history"][-1]["loss"] > 0
