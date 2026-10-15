import json
import os

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("onnxruntime")
from hae.detector import Detector
from hae.export import export_onnx
from hae.model.presets import build
from hae.results import Results


@pytest.fixture()
def models(tmp_path):
    root = tmp_path / "models"
    for name, mode, nc, tn in (("hae-n-ir", "lwir", 3, True), ("hae-n-rgb", "rgb", 1, False)):
        (root / name).mkdir(parents=True)
        export_onnx(build("n", ncls=nc, tnorm=tn).eval(), root / name / f"{name}_{mode}_int8.onnx", mode, (64, 96))
    return root


def _img(path, rgb):
    a = np.random.default_rng(1).integers(0, 255, (64, 96, 3) if rgb else (64, 96), np.uint8)
    Image.fromarray(a).save(path)
    return a


def test_results_object_has_yolo_like_accessors(models, tmp_path):
    a = _img(tmp_path / "x.png", False)
    r = Detector("hae-n-ir", root=models, conf=0.0).predict(a)           # one image -> one Results
    assert isinstance(r, Results) and len(r) == len(r.boxes) == len(r.conf) == len(r.cls)
    assert r.boxes.shape[1:] == (4,) and r.names[0] == "person" and r.orig_shape == (64, 96)
    img = r.plot()
    assert img.shape == (64, 96, 3) and img.dtype == np.uint8
    out = r.save(tmp_path / "pred.png")
    assert out.exists()
    data = json.loads(r.to_json())
    assert isinstance(data, list) and (not data or set(data[0]) == {"box", "score", "class"})
    assert "detections" in r.summary()


def test_model_by_name_and_list_or_folder_input(models, tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for k in range(3):
        _img(d / f"{k}.png", True)
    det = Detector("hae-n-rgb", root=models, conf=0.0)
    res = det.predict(str(d))                                              # folder -> list of Results sorted by name
    assert [os.path.basename(r.path) for r in res] == ["0.png", "1.png", "2.png"]
    assert len(det.predict([str(d / "0.png"), str(d / "1.png")])) == 2
    with pytest.raises(FileNotFoundError, match="hae-n-ir|Available"):
        Detector("hae-zzz", root=models)


def test_default_conf_comes_from_model_metadata_when_not_given(models):
    from hae.infer import embed_metadata
    p = models / "hae-n-ir" / "hae-n-ir_lwir_int8.onnx"
    embed_metadata(p, {"recommended_conf": 0.12})
    assert Detector("hae-n-ir", root=models).conf == 0.12
    assert Detector("hae-n-ir", root=models, conf=0.4).conf == 0.4
    assert Detector(p).conf == 0.12


def test_preset_selects_threshold_from_metadata(models):
    from hae.infer import embed_metadata
    p = models / "hae-n-ir" / "hae-n-ir_lwir_int8.onnx"
    embed_metadata(p, {"recommended_conf": 0.3, "sensitive_conf": 0.1})
    assert Detector("hae-n-ir", root=models).conf == 0.3
    assert Detector("hae-n-ir", root=models, preset="sensitive").conf == 0.1
    assert Detector("hae-n-ir", root=models, preset="sensitive", conf=0.2).conf == 0.2      # an explicit threshold wins over the preset
    with pytest.raises(ValueError):
        Detector("hae-n-ir", root=models, preset="weird")
