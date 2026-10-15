import json

import numpy as np
import pytest
from PIL import Image

from hae.data.annotations import load_annotations
from hae.data.cache import Cache, build_cache_thermal
from hae.data.extra import convert_airesq, convert_hituav
from hae.data.sampler import CropSampler, MixedSampler

H, W = 64, 96


def _img(path, v=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((H, W), v, np.uint8)).save(path, quality=100)


def _fake_hituav(root):
    nj = root / "normal_json"
    cats = [{"id": i, "name": n} for i, n in enumerate(["Person", "Car", "Bicycle", "OtherVehicle", "DontCare"])]
    for sp in ("train", "val", "test"):
        _img(nj / sp / f"{sp}_a.jpg")
        ann = [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 8, 12]},        # person
               {"id": 1, "image_id": 0, "category_id": 1, "bbox": [30, 20, 20, 10]},       # car -> background
               {"id": 2, "image_id": 0, "category_id": 2, "bbox": [50, 30, 9, 9]},         # bicycle -> ignore
               {"id": 3, "image_id": 0, "category_id": 4, "bbox": [60, 40, 9, 9]}]         # DontCare -> ignore
        (nj / "annotations").mkdir(parents=True, exist_ok=True)
        (nj / "annotations" / f"{sp}.json").write_text(json.dumps({
            "info": {}, "licenses": [], "categories": cats,
            "images": [{"filename": f"{sp}_a.jpg", "height": H, "width": W, "id": 0}], "annotation": ann}))


def test_convert_hituav_maps_classes_and_prefixes_names(tmp_path):
    src, out = tmp_path / "hit", tmp_path / "out"
    _fake_hituav(src)
    n = convert_hituav(src, out)
    assert n == 3
    fr = load_annotations(out / "ann.json")
    assert len(fr) == 3 and all(len(f.boxes) == 1 and len(f.ignore) == 2 for f in fr)   # only person is positive; cars dropped
    assert (out / "thermal" / fr[0].file_name).exists() and len({f.file_name for f in fr}) == 3


def test_convert_airesq_yolo_to_pixels_and_empty_labels(tmp_path):
    src, out = tmp_path / "ai", tmp_path / "out"
    _img(src / "images" / "a.JPG"); _img(src / "images" / "b.JPG")
    (src / "labels").mkdir()
    (src / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.2\n0 0.25 0.25 0.1 0.1\n")
    (src / "labels" / "b.txt").write_text("")
    assert convert_airesq(src, out, target_wh=(W, H)) == 2
    fr = {f.file_name: f for f in load_annotations(out / "ann.json")}
    a = fr["airesq_a.jpg"]
    np.testing.assert_allclose(a.boxes[0], [W * 0.45, H * 0.4, W * 0.55, H * 0.6], atol=0.01)
    assert len(fr["airesq_b.jpg"].boxes) == 0


def test_thermal_only_cache_and_sampler_modes(tmp_path):
    src, out, cache_dir = tmp_path / "hit", tmp_path / "out", tmp_path / "cache"
    _fake_hituav(src)
    convert_hituav(src, out)
    build_cache_thermal(out, "hituav", cache_dir)
    c = Cache(cache_dir, "hituav")
    assert c.rgb is None and c.t.shape == (3, H, W) and (c.n, c.h, c.w) == (3, H, W)
    b = CropSampler(c, load_annotations(out / "ann.json"), crop_hw=(32, 48), seed=0).sample_batch(4)
    assert b.modes == ("lwir",) and b.t.shape == (4, 1, 32, 48) and float(b.rgb.abs().max()) == 0.0


def test_mixed_sampler_respects_weights_and_modes(tmp_path):
    class Fake:
        def __init__(self, tag, modes): self.tag, self.modes = tag, modes
        def sample_batch(self, bs):
            from hae.data.sampler import Batch
            import torch
            return Batch(torch.zeros(bs, 3, 4, 4), torch.zeros(bs, 1, 4, 4), [], [], [], self.modes)
    m = MixedSampler([Fake("a", ("both", "rgb", "lwir")), Fake("b", ("lwir",))], [3.0, 1.0], seed=0)
    modes = [m.sample_batch(2).modes for _ in range(2000)]
    frac_b = sum(x == ("lwir",) for x in modes) / 2000
    assert 0.20 < frac_b < 0.30
    with pytest.raises(ValueError):
        MixedSampler([Fake("a", ("both",))], [1.0, 2.0])


def test_convert_airesq_downscales_larger_frames_to_target_and_keeps_relative_boxes(tmp_path):
    src, out = tmp_path / "ai", tmp_path / "out"
    big = src / "images" / "big.JPG"
    big.parent.mkdir(parents=True)
    Image.fromarray(np.full((2 * H, 2 * W), 90, np.uint8)).save(big, quality=100)
    _img(src / "images" / "small.JPG")
    (src / "labels").mkdir()
    (src / "labels" / "big.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    (src / "labels" / "small.txt").write_text("0 0.5 0.5 0.1 0.2\n")
    assert convert_airesq(src, out, target_wh=(W, H)) == 2
    for f in load_annotations(out / "ann.json"):
        assert (f.width, f.height) == (W, H)
        np.testing.assert_allclose(f.boxes[0], [W * 0.45, H * 0.4, W * 0.55, H * 0.6], atol=0.01)
        assert Image.open(out / "thermal" / f.file_name).size == (W, H)
