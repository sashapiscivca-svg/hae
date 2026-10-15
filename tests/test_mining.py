import json

import numpy as np
from PIL import Image

from hae.data.annotations import FrameAnn
from hae.data.cache import Cache, build_cache
from hae.data.sampler import CropSampler
from hae.mining import mine_from_detections
from hae.train import TrainConfig, train


def _ann(i, boxes, ignore=()):
    return FrameAnn(i, f"{i}.jpg", 64, 96, np.array(boxes, np.float32).reshape(-1, 4), np.array(ignore, np.float32).reshape(-1, 4),
                    np.zeros(len(boxes), np.int64))


def test_false_positives_exclude_matched_and_ignored_and_low_score():
    anns = [_ann(0, [[10, 10, 18, 18]], ignore=[[40, 40, 60, 60]]), _ann(1, [])]
    dets = [np.array([[10, 10, 18, 18, 0.9, 0],       # matches a person -> not a negative
                      [30, 5, 38, 13, 0.8, 0],        # false positive
                      [45, 45, 55, 55, 0.9, 0],       # inside an ignore region -> not a negative
                      [70, 5, 78, 13, 0.05, 0]], np.float32),   # low confidence
            np.array([[20, 20, 28, 28, 0.6, 0]], np.float32)]
    pts = mine_from_detections(dets, anns, [0, 1], score_thr=0.2)
    assert pts.shape == (2, 4) and set(pts[:, 0]) == {0.0, 1.0}
    assert np.allclose(pts[pts[:, 0] == 0][0, 1:3], [34, 9])


def test_sampler_centers_crops_on_hard_negatives_when_frac_is_one(tmp_path):
    root = tmp_path / "ds"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs = [{"id": 0, "file_name": "0.jpg", "height": 64, "width": 96}]
    for m in ("visible", "thermal"):
        Image.fromarray(np.zeros((64, 96, 3), np.uint8)).save(root / "train" / m / "0.jpg")
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": [], "categories": []}))
    build_cache(root, "train", tmp_path / "c")
    s = CropSampler(Cache(tmp_path / "c", "train"), [_ann(0, [])], crop_hw=(32, 32), seed=0)
    s.set_hard_negatives(np.array([[0.0, 80.0, 50.0, 0.9]], np.float32), frac=1.0)
    for _ in range(20):
        i, x0, y0 = s._origin(32, 32)
        assert i == 0 and x0 <= 80 <= x0 + 32 and y0 <= 50 <= y0 + 32


def test_train_with_hard_negative_mining_runs(tmp_path):
    root = tmp_path / "ds"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(6):
        im = np.zeros((64, 96, 3), np.uint8)
        im[20:30, 6 + 12 * i:16 + 12 * i] = 255
        for m in ("visible", "thermal"):
            Image.fromarray(im).save(root / "train" / m / f"{i}.jpg", quality=100)
        imgs.append({"id": i, "file_name": f"{i}.jpg", "height": 64, "width": 96})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [6 + 12 * i, 20, 10, 10], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", tmp_path / "c")
    cfg = TrainConfig(cache_dir=str(tmp_path / "c"), ann_root=str(root), out_dir=str(tmp_path / "r"), epochs=2, crops_per_epoch=16, batch=8,
                      threads=2, crop_hw=(32, 64), size="n", classes=1, prefetch=0, hn_frac=0.5, hn_start_epoch=1, hn_frames=6, hn_score=0.0)
    res = train(cfg)
    assert res["history"][-1]["hn_points"] >= 0
