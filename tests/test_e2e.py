"""End-to-end smoke test on a tiny synthetic thermal dataset, through the same code paths as the public scripts:
prepare (convert + cache) -> train (scripts/train.py) -> export fp32 + int8 (scripts/export.py) -> evaluate (scripts/evaluate.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
pytest.importorskip("onnxruntime")
import evaluate as E  # noqa: E402
import export as X  # noqa: E402
import prepare_data as P  # noqa: E402
import train as T  # noqa: E402

H, W = 64, 96


def _fake_hituav(root, n=4):
    """HIT-UAV-like download: normal_json/{annotations/<split>.json, <split>/*.jpg}; one warm 'person' per frame."""
    nj = root / "normal_json"
    (nj / "annotations").mkdir(parents=True)
    cats = [{"id": 0, "name": "Person"}, {"id": 1, "name": "Car"}]
    rng = np.random.default_rng(0)
    for sp in ("train", "val", "test"):
        (nj / sp).mkdir()
        imgs, anns = [], []
        for i in range(n):
            im = rng.integers(20, 60, (H, W), dtype=np.uint8)
            x, y = 8 + 18 * (i % 4), 10 + 8 * (i % 3)
            im[y:y + 12, x:x + 6] = 220
            Image.fromarray(im).save(nj / sp / f"{sp}{i}.jpg", quality=95)
            imgs.append({"id": i, "filename": f"{sp}{i}.jpg", "height": H, "width": W})
            anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [x, y, 6, 12]})
        (nj / "annotations" / f"{sp}.json").write_text(json.dumps({"categories": cats, "images": imgs, "annotation": anns}))


def test_prepare_train_export_evaluate(tmp_path):
    _fake_hituav(tmp_path / "src")
    data = tmp_path / "data"
    rep = P.prepare("hituav", tmp_path / "src", data, do_align=False)
    assert rep["hituav_trainval"]["kept"] == 8 and rep["hituav_test"]["kept"] == 4
    recipe = {"name": "tiny-ir", "train_sets": [{"name": "hituav_trainval", "weight": 1.0}], "val_sets": ["hituav_test"], "eval_stride": 1,
              "train": {"size": "n", "mode_probs": [0, 0, 1], "eval_modes": ["lwir"], "crop_hw": [32, 64], "batch": 8, "crops_per_epoch": 16,
                        "tnorm": True, "ctx": True, "scale_range": [0.9, 1.2], "eval_scale": 1.0, "copy_paste": 0.3, "neg_decay": 1.0,
                        "quick_val": 2, "tb": False}}
    run = tmp_path / "run"
    res = T.run(recipe, data, run, epochs=1, threads=2, device="cpu")
    final = json.loads((run / "final.json").read_text())
    assert set(final) == {"lwir@hituav_test"} and "AP50" in final["lwir@hituav_test"] and res["cfg"].train_set == "hituav_trainval"
    rel = tmp_path / "rel"
    xr = X.export(run / "last.pt", rel, data, ["hituav_trainval"], name="tiny-ir", calib_n=4, conf=0.3, sensitive_conf=0.2, tile_hw=(64, 96))
    assert xr["mode"] == "lwir" and xr["torch_vs_onnx_fp32_max_abs_diff"] < 1e-3
    for f in ("tiny-ir_lwir.onnx", "tiny-ir_lwir_int8.onnx", "tiny-ir.safetensors", "config.json"):
        assert (rel / f).exists(), f
    for f in ("tiny-ir_lwir.onnx", "tiny-ir_lwir_int8.onnx"):
        er = E.evaluate_model(rel / f, data, ["hituav_test"])
        s = er["sets"]["hituav_test"]
        assert er["presets"] == {"balanced": 0.3, "sensitive": 0.2} and set(s["at"]) == {"0.3", "0.2"}
        assert s["frames"] == 4 and s["gt"] == 4 and 0.0 <= s["AP50"] <= 1.0
        assert {"recall", "precision", "f1", "fp_per_frame"} <= set(s["at"]["0.3"])


def test_finetune_from_released_weights(tmp_path):
    """--init: start from the released hae-s-ir safetensors; the recipe's architecture must match (strict load)."""
    st = ROOT / "models" / "hae-s-ir" / "hae-s-ir.safetensors"
    if not st.exists():
        pytest.skip("released model not present")
    _fake_hituav(tmp_path / "src")
    data = tmp_path / "data"
    P.prepare("hituav", tmp_path / "src", data, do_align=False)
    base = {"mode_probs": [0, 0, 1], "eval_modes": ["lwir"], "crop_hw": [64, 96], "batch": 4, "crops_per_epoch": 8, "tnorm": True, "ctx": True,
            "eval_scale": 1.0, "quick_val": 0, "tb": False}
    recipe = {"name": "ft", "train_sets": [{"name": "hituav_trainval", "weight": 1.0}], "val_sets": ["hituav_test"], "eval_stride": 1,
              "train": {**base, "size": "s"}}
    res = T.run(recipe, data, tmp_path / "ft", epochs=1, threads=2, device="cpu", init_weights=str(st))
    assert res["cfg"].init_weights == str(st) and (tmp_path / "ft" / "last.pt").exists()
    wrong = {**recipe, "train": {**base, "size": "n"}}
    with pytest.raises(RuntimeError):
        T.run(wrong, data, tmp_path / "ft-wrong", epochs=1, threads=2, device="cpu", init_weights=str(st))
