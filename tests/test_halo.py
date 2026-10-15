import numpy as np

from hae.data.halo import halo_mask


def _scene(h=64, w=96, peak=(30, 40), sigma=1.6, amp=120.0, base=60.0):
    ys, xs = np.mgrid[0:h, 0:w]
    img = base + amp * np.exp(-(((ys - peak[0]) ** 2 + (xs - peak[1]) ** 2) / (2 * sigma ** 2)))
    return np.clip(img + np.random.default_rng(0).normal(0, 2, img.shape), 0, 255).astype(np.uint8)


def test_mask_is_anchored_at_the_hotspot_not_the_box_centre():
    img = _scene(peak=(30, 44))
    shifted = np.array([[38, 20, 54, 36]], np.float32)       # centre (46,28), hot spot (44,30) - shift ~2.8 px
    m = halo_mask(img, shifted)
    py, px = np.unravel_index(m.argmax(), m.shape)
    assert abs(py - 30) <= 1 and abs(px - 44) <= 1
    assert 0.0 <= m.min() and m.max() <= 1.0 and m.max() > 0.9
    assert m[:15].max() == 0.0 and m[:, :20].max() == 0.0     # outside the box - zero


def test_multiple_boxes_combine_by_max_and_edge_boxes_are_safe():
    img = _scene(peak=(30, 44))
    img = np.maximum(img, _scene(peak=(10, 8)))
    boxes = np.array([[36, 22, 52, 38], [0, 2, 16, 18]], np.float32)
    m = halo_mask(img, boxes)
    assert m[30, 44] > 0.8 and m[10, 8] > 0.8
    assert halo_mask(img, np.zeros((0, 4), np.float32)).max() == 0.0


def test_flat_background_has_no_nan_and_degenerate_boxes_ignored():
    flat = np.full((64, 96), 100, np.uint8)
    m = halo_mask(flat, np.array([[10, 10, 24, 24], [30, 30, 30, 40]], np.float32))
    assert np.isfinite(m).all() and m.shape == (64, 96)


def test_cache_halo_sampler_alignment_flip_scale_and_copy_paste(tmp_path):
    import json
    from PIL import Image
    from hae.data.annotations import load_annotations
    from hae.data.cache import Cache, build_cache, build_halo_cache
    from hae.data.sampler import CropSampler
    H, W = 64, 96
    root, out = tmp_path / "ds", tmp_path / "cache"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(3):
        th = _scene(H, W, peak=(25, 50), sigma=1.5, amp=150, base=50)
        Image.fromarray(np.stack([th] * 3, -1)).save(root / "train" / "thermal" / f"{i:05d}.jpg", quality=100)
        Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(root / "train" / "visible" / f"{i:05d}.jpg")
        imgs.append({"id": i, "file_name": f"{i:05d}.jpg", "height": H, "width": W})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [43, 18, 14, 14], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", out)
    cache, fr = Cache(out, "train"), load_annotations(root / "train_thermal.json")
    assert cache.halo is None
    build_halo_cache(cache, fr, "train", out)
    cache = Cache(out, "train")
    assert cache.halo is not None and cache.halo.shape == (3, H, W)
    for kw in (dict(flip=True), dict(flip=False, scale_range=(1.5, 1.5)), dict(flip=True, copy_paste=1.0, cp_max=1)):
        s = CropSampler(cache, fr, crop_hw=(32, 48), pos_frac=1.0, seed=2, use_halo=True, **kw)
        b = s.sample_batch(6)
        assert b.halo is not None and b.halo.shape == (6, 1, 32, 48) and 0.0 <= float(b.halo.min()) and float(b.halo.max()) <= 1.0
        for k in range(6):
            for x1, y1, x2, y2 in b.boxes[k]:
                reg = b.halo[k, 0, int(y1):int(y2) + 1, int(x1):int(x2) + 1]
                assert float(reg.max()) > 0.5                      # the mask lies within its box after flips/scaling/paste
    assert CropSampler(cache, fr, crop_hw=(32, 48), seed=0).sample_batch(2).halo is None   # without use_halo the field is empty


def test_soft_iou_loss_properties():
    import torch
    from hae.loss import soft_iou_loss
    y = torch.zeros(2, 1, 8, 8); y[:, :, 3:5, 3:5] = 1.0
    perfect = (y * 20 - 10)                                   # logits: +10 where the mask is, -10 on background
    assert float(soft_iou_loss(perfect, y)) < 0.2
    assert float(soft_iou_loss(-perfect, y)) > 0.8
    assert float(soft_iou_loss(torch.zeros_like(y), torch.zeros_like(y))) < 1.0
