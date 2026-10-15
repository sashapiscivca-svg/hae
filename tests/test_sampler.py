import json

import numpy as np
import pytest
from PIL import Image

from hae.data.annotations import load_annotations
from hae.data.cache import Cache, build_cache
from hae.data.sampler import CropSampler, load_frames

H, W = 64, 96


def _dataset(tmp_path, with_target=True):
    root, out = tmp_path / "ds", tmp_path / "cache"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(3):
        im = np.zeros((H, W, 3), np.uint8)
        if with_target and i < 2:
            im[20:30, 40:50] = 255  # bright 10x10 square
        for m in ("visible", "thermal"):
            Image.fromarray(im).save(root / "train" / m / f"{i:05d}.jpg", quality=100)
        imgs.append({"id": i, "file_name": f"{i:05d}.jpg", "height": H, "width": W})
        if with_target and i < 2:
            anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [40, 20, 10, 10], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", out)
    return Cache(out, "train"), load_annotations(root / "train_thermal.json")


def test_shapes_and_range(tmp_path):
    cache, anns = _dataset(tmp_path)
    b = CropSampler(cache, anns, crop_hw=(32, 48), seed=1).sample_batch(8)
    assert b.rgb.shape == (8, 3, 32, 48) and b.t.shape == (8, 1, 32, 48)
    assert 0.0 <= float(b.rgb.min()) and float(b.rgb.max()) <= 1.0
    assert len(b.boxes) == 8 and len(b.ignore) == 8


def test_positive_crops_contain_target_and_flips_keep_alignment(tmp_path):
    cache, anns = _dataset(tmp_path)
    s = CropSampler(cache, anns, crop_hw=(32, 48), pos_frac=1.0, seed=3, flip=True)
    seen = 0
    for _ in range(20):
        b = s.sample_batch(4)
        for k in range(4):
            assert len(b.boxes[k]) >= 1 or len(b.ignore[k]) >= 1
            for x1, y1, x2, y2 in b.boxes[k]:
                assert 0 <= x1 < x2 <= 48 and 0 <= y1 < y2 <= 32
                region = b.rgb[k, :, int(y1):int(y2), int(x1):int(x2)]
                assert float(region.mean()) > 0.9  # the box really covers the bright square after flips
                assert float(b.t[k, :, int(y1):int(y2), int(x1):int(x2)].mean()) > 0.9
                seen += 1
    assert seen > 0


def test_partially_visible_box_becomes_ignore(tmp_path):
    cache, anns = _dataset(tmp_path)
    s = CropSampler(cache, anns, crop_hw=(32, 44), pos_frac=0.0, seed=0, flip=False)
    # crop x in [0,44) cuts box [40,20,50,30]: 4 of 10 px visible = 40 % < 50 % -> ignore
    boxes, ignore, _ = s._boxes_for_crop(anns[0], x0=0, y0=0, flipx=False, flipy=False)
    assert len(boxes) == 0 and len(ignore) == 1
    # crop x in [30,74), y in [10,42) fully covers the box -> positive
    boxes, ignore, _ = s._boxes_for_crop(anns[0], x0=30, y0=10, flipx=False, flipy=False)
    assert len(boxes) == 1 and len(ignore) == 0


def test_frames_without_targets_are_fine(tmp_path):
    cache, anns = _dataset(tmp_path, with_target=False)
    b = CropSampler(cache, anns, crop_hw=(32, 48), pos_frac=0.7, seed=0).sample_batch(4)
    assert all(len(x) == 0 for x in b.boxes)


def test_frames_subset_and_load_frames(tmp_path):
    cache, anns = _dataset(tmp_path)
    s = CropSampler(cache, anns, crop_hw=(32, 48), seed=0, frames=[1])
    for _ in range(5):
        s.sample_batch(2)
    rgb, t = load_frames(cache, [0, 2])
    assert rgb.shape == (2, 3, H, W) and t.shape == (2, 1, H, W)


def test_scale_aug_keeps_shape_and_box_alignment(tmp_path):
    cache, anns = _dataset(tmp_path)
    for lo, hi in ((2.0, 2.0), (0.5, 0.5), (0.5, 2.0)):
        s = CropSampler(cache, anns, crop_hw=(32, 48), pos_frac=1.0, seed=5, flip=True, scale_range=(lo, hi))
        seen = 0
        for _ in range(15):
            b = s.sample_batch(4)
            assert b.rgb.shape == (4, 3, 32, 48) and b.t.shape == (4, 1, 32, 48)
            for k in range(4):
                for x1, y1, x2, y2 in b.boxes[k]:
                    assert 0 <= x1 < x2 <= 48 and 0 <= y1 < y2 <= 32
                    inside = x1 > 0 and y1 > 0 and x2 < 48 and y2 < 32   # boxes clipped by the border are smaller
                    if lo == hi and inside:   # fixed scale: a 10 px box becomes 10*s px
                        assert abs((x2 - x1) - 10 * lo) < 1e-3
                    if lo == hi:
                        assert (x2 - x1) <= 10 * lo + 1e-3
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    assert float(b.rgb[k, :, cy, cx].mean()) > 0.8 and float(b.t[k, :, cy, cx].mean()) > 0.8
                    seen += 1
        assert seen > 0


def test_scale_range_default_is_identity_and_invalid_range_raises(tmp_path):
    cache, anns = _dataset(tmp_path)
    a = CropSampler(cache, anns, crop_hw=(32, 48), seed=7).sample_batch(4)
    b = CropSampler(cache, anns, crop_hw=(32, 48), seed=7, scale_range=(1.0, 1.0)).sample_batch(4)
    assert (a.rgb == b.rgb).all()
    with pytest.raises(ValueError):
        CropSampler(cache, anns, crop_hw=(32, 48), scale_range=(2.0, 1.0))


def test_labels_follow_boxes_through_crop_filter_and_flip(tmp_path):
    from hae.data.annotations import FrameAnn
    cache, anns = _dataset(tmp_path)
    ann = FrameAnn(0, "x", H, W, np.array([[40, 20, 50, 30], [10, 10, 20, 20], [60, 40, 70, 50]], np.float32),
                   np.zeros((0, 4), np.float32), np.array([2, 1, 0], np.int64))
    s = CropSampler(cache, anns, crop_hw=(32, 44), pos_frac=0.0, seed=0, flip=False)
    boxes, ign, labels = s._boxes_for_crop(ann, x0=0, y0=0, flipx=False, flipy=False)
    assert labels.tolist() == [1] and len(boxes) == 1 and len(ign) == 1     # first box clipped -> ignore; third outside the crop
    s2 = CropSampler(cache, anns, crop_hw=(32, 48), pos_frac=1.0, seed=1, flip=True)
    b = s2.sample_batch(6)
    assert len(b.labels) == 6 and all(len(l) == len(bx) for l, bx in zip(b.labels, b.boxes))
    assert all(l.dtype == np.int64 for l in b.labels)


def test_copy_paste_adds_aligned_boxes_with_feathered_patches(tmp_path):
    cache, anns = _dataset(tmp_path)
    plain = CropSampler(cache, anns, crop_hw=(32, 48), pos_frac=1.0, seed=3, flip=False)
    paste = CropSampler(cache, anns, crop_hw=(32, 48), pos_frac=1.0, seed=3, flip=False, copy_paste=1.0, cp_max=2)
    more = 0
    for _ in range(12):
        a, b = plain.sample_batch(4), paste.sample_batch(4)
        for k in range(4):
            more += len(b.boxes[k]) - len(a.boxes[k])
            assert len(b.labels[k]) == len(b.boxes[k])
            for x1, y1, x2, y2 in b.boxes[k]:
                assert 0 <= x1 < x2 <= 48 and 0 <= y1 < y2 <= 32
                cy, cx = int((y1 + y2) / 2), int((x1 + x2) / 2)
                assert float(b.t[k, 0, cy, cx]) > 0.8        # the pasted target is really bright at its place
    assert more > 0


def test_copy_paste_disabled_by_default_and_validates_probability(tmp_path):
    cache, anns = _dataset(tmp_path)
    a = CropSampler(cache, anns, crop_hw=(32, 48), seed=9).sample_batch(4)
    b = CropSampler(cache, anns, crop_hw=(32, 48), seed=9, copy_paste=0.0).sample_batch(4)
    assert (a.t == b.t).all()
    with pytest.raises(ValueError):
        CropSampler(cache, anns, crop_hw=(32, 48), copy_paste=1.5)


def test_rare_class_crops_are_centred_on_rare_instances(tmp_path):
    import json as _json
    from hae.data.annotations import load_annotations as _load
    root, out = tmp_path / "ds", tmp_path / "cache"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(3):
        im = np.zeros((H, W, 3), np.uint8)
        im[10:20, 10:20] = 255      # person (class 0)
        im[40:50, 70:80] = 200      # rider  (class 1)
        for m in ("visible", "thermal"):
            Image.fromarray(im).save(root / "train" / m / f"{i:05d}.jpg", quality=100)
        imgs.append({"id": i, "file_name": f"{i:05d}.jpg", "height": H, "width": W})
        anns += [{"id": 2 * i, "image_id": i, "category_id": 0, "bbox": [10, 10, 10, 10], "ignore": 0, "iscrowd": 0},
                 {"id": 2 * i + 1, "image_id": i, "category_id": 1, "bbox": [70, 40, 10, 10], "ignore": 0, "iscrowd": 0}]
    (root / "train_thermal.json").write_text(_json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", out)
    cache, fr = Cache(out, "train"), _load(root / "train_thermal.json", mode="coco3")
    s = CropSampler(cache, fr, crop_hw=(32, 48), pos_frac=1.0, seed=0, flip=False, rare_crop=1.0)
    has_rider = 0
    for _ in range(20):
        b = s.sample_batch(4)
        has_rider += sum(int((l == 1).any()) for l in b.labels)
    assert has_rider == 80                                         # every crop contains a rider
    base = CropSampler(cache, fr, crop_hw=(32, 48), pos_frac=1.0, seed=0, flip=False)
    n_plain = sum(int((l == 1).any()) for _ in range(20) for l in base.sample_batch(4).labels)
    assert n_plain < 80
