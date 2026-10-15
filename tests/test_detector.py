import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
from hae.data.cache import Cache, build_cache
from hae.detector import Detector
from hae.export import export_onnx
from hae.infer import embed_metadata, preprocess_rgb, preprocess_thermal, read_metadata
from hae.model.presets import build
from hae.evaluation import detect_onnx


def _frame(h=64, w=96, seed=0):
    r = np.random.default_rng(seed)
    t = r.integers(40, 90, (h, w), np.uint8)
    t[20:28, 30:38] = 230          # warm spot
    return t


def test_thermal_preprocess_matches_training_formula():
    t = _frame()
    x = preprocess_thermal(t, tnorm=True)
    ref = (t.astype(np.float32) - t.mean()) / max(float(t.std()), 1.0)
    assert x.shape == (1, 1, 64, 96) and x.dtype == np.float32 and np.allclose(x[0, 0], ref, atol=1e-4)
    assert np.allclose(preprocess_thermal(t, tnorm=False)[0, 0], t / 255.0)
    assert preprocess_rgb(np.zeros((8, 8, 3), np.uint8)).shape == (1, 3, 8, 8)


def test_metadata_roundtrip_survives_embedding(tmp_path):
    m = build("n", ncls=3, tnorm=True).eval()
    p = tmp_path / "m.onnx"
    export_onnx(m, p, "lwir", (64, 96))
    meta = read_metadata(p)
    assert meta["mode"] == "lwir" and meta["tnorm"] is True and meta["stride"] == 4 and meta["ncls"] == 3
    embed_metadata(p, {"name": "hae-test", "recommended_scale": 1.5})
    meta2 = read_metadata(p)
    assert meta2["name"] == "hae-test" and meta2["recommended_scale"] == 1.5 and meta2["mode"] == "lwir"   # existing fields are kept


def test_detector_matches_reference_pipeline_and_does_not_import_torch(tmp_path):
    m = build("n", ncls=3, tnorm=True).eval()
    p = tmp_path / "m.onnx"
    export_onnx(m, p, "lwir", (64, 96))
    t = _frame()
    (tmp_path / "d" / "train" / "visible").mkdir(parents=True)
    (tmp_path / "d" / "train" / "thermal").mkdir(parents=True)
    Image.fromarray(np.stack([t] * 3, -1)).save(tmp_path / "d" / "train" / "visible" / "0.png")
    Image.fromarray(t).save(tmp_path / "d" / "train" / "thermal" / "0.png")
    (tmp_path / "d" / "train_thermal.json").write_text(json.dumps({"images": [{"id": 0, "file_name": "0.png", "height": 64, "width": 96}], "annotations": [], "categories": []}))
    build_cache(tmp_path / "d", "train", tmp_path / "c")
    ref = detect_onnx(p, Cache(tmp_path / "c", "train"), [0], "lwir", True, score_thr=0.0)[0]
    det = Detector(p, conf=0.0)
    out = det.detect(t)
    assert out.shape[1] == 6 and len(out) > 0
    top_ref, top = ref[np.argsort(-ref[:, 4])][:5], out[np.argsort(-out[:, 4])][:5]
    assert np.allclose(top_ref[:, :5], top[:, :5], atol=1e-2)
    code = "import sys, hae.detector; assert 'torch' not in sys.modules, 'detector imports torch'"
    assert subprocess.run([sys.executable, "-c", code], env={"PYTHONPATH": str(__import__('pathlib').Path(__file__).parents[1] / 'src'), "PATH": ""}).returncode == 0


def test_scale_tiling_and_input_validation(tmp_path):
    m = build("n", ncls=1).eval()
    p = tmp_path / "m.onnx"
    export_onnx(m, p, "rgb", (64, 96))
    img = np.random.default_rng(0).integers(0, 255, (90, 130, 3), np.uint8)           # size not a multiple of 32
    full = Detector(p, conf=0.0).detect(img)
    scaled = Detector(p, conf=0.0, scale=1.5).detect(img)
    tiled = Detector(p, conf=0.0, tile=(64, 96), overlap=0.25).detect(img)
    for o in (full, scaled, tiled):
        assert o.shape[1] >= 5 and (o[:, [0, 2]] <= 130 + 1e-3).all() and (o[:, [1, 3]] <= 90 + 1e-3).all()
    with pytest.raises(ValueError):
        Detector(p).detect(np.zeros((64, 96), np.uint8))                              # an rgb model expects (H,W,3)


def test_flip_and_scale_override_are_consistent(tmp_path):
    m = build("n", ncls=1).eval()
    p = tmp_path / "m.onnx"
    export_onnx(m, p, "rgb", (64, 96))
    img = np.random.default_rng(3).integers(0, 255, (64, 96, 3), np.uint8)
    d = Detector(p, conf=0.0)
    base = d.detect(img)
    flipped = d.detect(img, flip=True)
    mirrored = base.copy()
    mirrored[:, [0, 2]] = 96 - base[:, [2, 0]]
    assert flipped.shape[1] == 6 and (flipped[:, [0, 2]] >= 0).all() and (flipped[:, [0, 2]] <= 96).all()
    assert not np.allclose(np.sort(flipped[:, 0]), np.sort(base[:, 0]))               # mirroring does change the detections
    assert np.array_equal(d.detect(img, scale=1.0), base)                              # explicit scale = default
    assert not np.array_equal(d.detect(img, scale=2.0), base) and d.scale == 1.0       # the override does not change the object's setting
    assert np.allclose(np.sort(mirrored[:, 4]), np.sort(base[:, 4]))

