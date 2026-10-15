import json

import numpy as np
from PIL import Image

from hae.data.cache import Cache, build_cache_rgb


def _rgb_set(d, n=3, h=112, w=208):
    (d / "visible").mkdir(parents=True)
    for k in range(n):
        Image.fromarray(np.random.default_rng(k).integers(0, 255, (h, w, 3), np.uint8)).save(d / "visible" / f"f{k}.png")
    (d / "ann.json").write_text(json.dumps({"images": [{"id": k, "file_name": f"f{k}.png", "height": h, "width": w} for k in range(n)],
                                            "annotations": [{"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 8, 8]}], "categories": []}))


def test_rgb_only_cache_uses_luma_for_thermal_and_restricts_modes(tmp_path):
    _rgb_set(tmp_path / "s")
    build_cache_rgb(tmp_path / "s", "rgbset", tmp_path / "cache")
    c = Cache(tmp_path / "cache", "rgbset")
    assert c.rgb is not None and c.t.shape == c.rgb.shape[:3]
    assert abs(float(c.t[0].mean()) - float(c.rgb[0].astype(np.float32).mean(-1).mean())) < 6   # t = RGB luminance
    assert json.loads((tmp_path / "cache" / "rgbset_meta.json").read_text())["modes"] == ["rgb"]


def test_allowed_modes_restricts_only_full_modality_batches():
    from hae.train import allowed_modes
    full = ("both", "rgb", "lwir")
    assert allowed_modes(full, ("both", "lwir")) == ["both", "lwir"]     # main RGBT set: rgb labels unreliable
    assert allowed_modes(("rgb",), ("both", "lwir")) == ["rgb"]          # RGB sets stay rgb
    assert allowed_modes(("lwir",), ("both", "lwir")) == ["lwir"]
    assert allowed_modes(full, None) == ["both", "rgb", "lwir"]


def test_load_frames_works_on_thermal_only_cache(tmp_path):
    from hae.data.cache import build_cache_thermal
    from hae.data.sampler import load_frames
    src = tmp_path / "u"
    (src / "thermal").mkdir(parents=True)
    for i in range(3):
        Image.fromarray(np.full((32, 48), 50 + 10 * i, np.uint8)).save(src / "thermal" / f"t{i}.jpg", quality=100)
    (src / "ann.json").write_text(json.dumps({"images": [{"id": i, "file_name": f"t{i}.jpg", "height": 32, "width": 48} for i in range(3)],
                                              "annotations": [], "categories": []}))
    build_cache_thermal(src, "thm", tmp_path / "c")
    rgb, t = load_frames(Cache(tmp_path / "c", "thm"), [0, 2], t_norm=False)
    assert rgb.shape == (2, 3, 32, 48) and t.shape == (2, 1, 32, 48)
    assert float(t[1].mean()) > float(t[0].mean())
