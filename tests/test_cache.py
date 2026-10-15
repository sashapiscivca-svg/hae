import json

import numpy as np
import pytest
from PIL import Image

from hae.data.cache import Cache, build_cache


def _make_dataset(root, n=3, h=32, w=48):
    (root / "train" / "visible").mkdir(parents=True)
    (root / "train" / "thermal").mkdir(parents=True)
    images, anns = [], []
    for i in range(n):
        name = f"{i:05d}.jpg"
        Image.fromarray(np.full((h, w, 3), 40 * (i + 1), np.uint8)).save(root / "train" / "visible" / name, quality=100)
        Image.fromarray(np.full((h, w, 3), 90 + 20 * i, np.uint8)).save(root / "train" / "thermal" / name, quality=100)
        images.append({"id": i, "file_name": name, "height": h, "width": w})
    anns.append({"id": 0, "image_id": 1, "category_id": 0, "bbox": [4, 4, 8, 8], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": []}))


def test_build_and_read(tmp_path):
    root, out = tmp_path / "ds", tmp_path / "cache"
    _make_dataset(root)
    build_cache(root, "train", out)
    c = Cache(out, "train")
    assert (c.n, c.h, c.w) == (3, 32, 48)
    assert c.rgb.shape == (3, 32, 48, 3) and c.t.shape == (3, 32, 48)
    assert abs(int(c.rgb[1].mean()) - 80) <= 2
    assert abs(int(c.t[2].mean()) - 130) <= 2


def test_size_mismatch_raises(tmp_path):
    root, out = tmp_path / "ds", tmp_path / "cache"
    _make_dataset(root)
    Image.fromarray(np.zeros((10, 10, 3), np.uint8)).save(root / "train" / "thermal" / "00001.jpg")
    with pytest.raises(ValueError, match="size"):
        build_cache(root, "train", out)


def test_build_cache_reuses_existing_cache_unless_forced(tmp_path):
    import json

    import numpy as np
    from PIL import Image

    from hae.data.cache import build_cache
    root = tmp_path / "d"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
        Image.fromarray(np.full((32, 48, 3), 90, np.uint8)).save(root / "train" / m / "0.jpg", quality=100)
    (root / "train_thermal.json").write_text(json.dumps({"images": [{"id": 0, "file_name": "0.jpg", "height": 32, "width": 48}], "annotations": [], "categories": []}))
    build_cache(root, "train", tmp_path / "c")
    stamp = (tmp_path / "c" / "train_meta.json").stat().st_mtime_ns
    build_cache(root, "train", tmp_path / "c")                       # second call does not rebuild
    assert (tmp_path / "c" / "train_meta.json").stat().st_mtime_ns == stamp
    build_cache(root, "train", tmp_path / "c", reuse=False)            # forced rebuild
    assert (tmp_path / "c" / "train_meta.json").stat().st_mtime_ns != stamp


def test_build_cache_does_not_reuse_cache_of_another_source(tmp_path):
    """Regression: a `val` cache in a shared directory built from one dataset must not be reused for another dataset (frame indices would not match the labels)."""
    import json

    import numpy as np
    from PIL import Image

    from hae.data.cache import build_cache

    def make(root, n):
        imgs = []
        for i in range(n):
            for m in ("visible", "thermal"):
                (root / "val" / m).mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((32, 48, 3), 60 + i, np.uint8)).save(root / "val" / m / f"{i}.jpg", quality=100)
            imgs.append({"id": i, "file_name": f"{i}.jpg", "height": 32, "width": 48})
        (root / "val_thermal.json").write_text(json.dumps({"images": imgs, "annotations": [], "categories": []}))

    a, b, out = tmp_path / "A", tmp_path / "B", tmp_path / "cache"
    make(a, 1)
    make(b, 3)
    build_cache(a, "val", out)
    assert json.loads((out / "val_meta.json").read_text())["n"] == 1
    build_cache(b, "val", out)                                  # another dataset with the same split name
    assert json.loads((out / "val_meta.json").read_text())["n"] == 3
    stamp = (out / "val_meta.json").stat().st_mtime_ns
    build_cache(b, "val", out)                                  # the same dataset: reused
    assert (out / "val_meta.json").stat().st_mtime_ns == stamp
