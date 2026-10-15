import json

import numpy as np
import torch
from PIL import Image

from hae.data.annotations import load_annotations
from hae.data.cache import Cache, build_cache
from hae.data.sampler import CropSampler, load_frames
from hae.model.hae import HaeConfig
from hae.model.presets import build

H, W = 64, 96


def _make(tmp_path):
    """Two frames with the same pattern but different thermal gain (brightness/contrast)."""
    root, out = tmp_path / "ds", tmp_path / "cache"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, (H, W))
    imgs = []
    for i, (gain, off) in enumerate([(1.0, 0), (0.5, 90)]):
        th = np.clip(base * 200 * gain + off, 0, 255).astype(np.uint8)
        Image.fromarray(np.stack([th] * 3, -1)).save(root / "train" / "thermal" / f"{i:05d}.jpg", quality=100)
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(root / "train" / "visible" / f"{i:05d}.jpg")
        imgs.append({"id": i, "file_name": f"{i:05d}.jpg", "height": H, "width": W})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": [], "categories": []}))
    build_cache(root, "train", out)
    return Cache(out, "train"), load_annotations(root / "train_thermal.json")


def test_frame_stats_are_cached_and_correct(tmp_path):
    c, _ = _make(tmp_path)
    mean, std = c.tstats
    assert mean.shape == (2,) and std.shape == (2,)
    assert abs(mean[0] - np.asarray(c.t[0]).mean()) < 1e-3 and abs(std[1] - np.asarray(c.t[1]).std()) < 1e-3
    assert (tmp_path / "cache" / "train_tstats.npy").exists()


def test_normalised_frames_remove_gain_and_offset_differences(tmp_path):
    c, anns = _make(tmp_path)
    _, t_norm = load_frames(c, [0, 1], t_norm=True)
    assert torch.allclose(t_norm[0], t_norm[1], atol=0.12)           # different gain -> almost the same input
    _, t_raw = load_frames(c, [0, 1], t_norm=False)
    assert not torch.allclose(t_raw[0], t_raw[1], atol=0.12)
    s = CropSampler(c, anns, crop_hw=(32, 48), seed=0, flip=False, t_norm=True).sample_batch(8)
    assert float(s.t.mean().abs()) < 0.5 and 0.5 < float(s.t.std()) < 1.6   # z-scores, not 0..1


def test_model_config_flag_and_build():
    assert HaeConfig().tnorm is False and HaeConfig(tnorm=True).scaled(2.0, 1.0).tnorm is True
    assert build(size="n", tnorm=True).cfg.tnorm is True
