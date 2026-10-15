import json

import numpy as np
import pytest
import torch
from PIL import Image

from hae.checkpoint import load_checkpoint
from hae.data.annotations import load_annotations
from hae.data.cache import Cache, build_cache
from hae.evaluation import eval_onnx, prf_table, release_model
from hae.train import TrainConfig, train

ort = pytest.importorskip("onnxruntime")


def _tiny(tmp_path, n=6, h=64, w=96):
    root, out = tmp_path / "ds", tmp_path / "cache"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(n):
        im = np.zeros((h, w, 3), np.uint8)
        x = 6 + 12 * (i % 6)
        im[20:30, x:x + 10] = 255
        for m in ("visible", "thermal"):
            Image.fromarray(im).save(root / "train" / m / f"{i:05d}.jpg", quality=100)
        imgs.append({"id": i, "file_name": f"{i:05d}.jpg", "height": h, "width": w})
        anns.append({"id": i, "image_id": i, "category_id": i % 3, "bbox": [x, 20, 10, 10], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", out)
    return root, out


def _train(tmp_path, root, cache_dir, **kw):
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1, crops_per_epoch=16,
                      batch=8, threads=2, crop_hw=(32, 64), size="n", **{"classes": 3, "tnorm": True, **kw})
    return train(cfg)


def test_checkpoint_roundtrip_pt_and_safetensors(tmp_path):
    from hae.checkpoint import save_safetensors
    root, cache_dir = _tiny(tmp_path)
    res = _train(tmp_path, root, cache_dir)
    model, cfg = load_checkpoint(tmp_path / "run" / "last.pt")
    assert cfg["size"] == "n" and model.cfg.ncls == 3 and model.cfg.tnorm is True
    x = (torch.rand(1, 3, 64, 96), torch.rand(1, 1, 64, 96))
    with torch.no_grad():
        a, b = res["model"](rgb=x[0], t=x[1]), model(rgb=x[0], t=x[1])
    assert torch.allclose(a["cls"], b["cls"], atol=1e-5)
    p = save_safetensors(model, cfg, tmp_path / "st", "tiny")
    conf = json.loads((tmp_path / "st" / "config.json").read_text())
    assert "cache_dir" not in conf["training"] and conf["architecture"]["ncls"] == 3
    m2, cfg2 = load_checkpoint(p)
    with torch.no_grad():
        assert torch.allclose(m2(t=x[1])["cls"], model(t=x[1])["cls"], atol=1e-6)


@pytest.mark.parametrize("mode", ["rgb", "lwir"])
def test_release_model_single_mode_bundle(tmp_path, mode):
    from hae.infer import read_metadata
    root, cache_dir = _tiny(tmp_path)
    _train(tmp_path, root, cache_dir, tnorm=(mode == "lwir"))
    c = Cache(cache_dir, "train")
    rep = release_model(tmp_path / "run" / "last.pt", tmp_path / "rel", name="n_x", mode=mode, calib_caches=[c], calib_n=4, tile_hw=(64, 96),
                        conf=0.4, sensitive_conf=0.3)
    for f in (f"n_x_{mode}.onnx", f"n_x_{mode}_int8.onnx", "n_x.safetensors", "config.json", "report.json"):
        assert (tmp_path / "rel" / f).exists(), f
    assert rep["mode"] == mode and rep["torch_vs_onnx_fp32_max_abs_diff"] < 1e-3
    assert rep["files"][f"n_x_{mode}.onnx"]["bytes"] > rep["files"][f"n_x_{mode}_int8.onnx"]["bytes"]
    m8 = read_metadata(tmp_path / "rel" / f"n_x_{mode}_int8.onnx")
    assert m8["mode"] == mode and m8["recommended_conf"] == 0.4 and m8["sensitive_conf"] == 0.3 and m8["precision"].startswith("int8")
    assert read_metadata(tmp_path / "rel" / f"n_x_{mode}.onnx")["precision"] == "fp32"


def test_int8_keeps_output_head_in_fp32(tmp_path):
    import onnx
    from hae.export import export_onnx
    from hae.model.presets import build
    from hae.quantize import quantize_int8
    fp, q = tmp_path / "m.onnx", tmp_path / "m_int8.onnx"
    export_onnx(build("n"), fp, "rgb", (64, 96))
    quantize_int8(fp, q, {"rgb": [np.random.rand(1, 3, 64, 96).astype(np.float32) for _ in range(4)]})
    g = onnx.load(str(q)).graph
    producer = {o: n for n in g.node for o in n.output}
    heads = [n for n in g.node if n.op_type == "Conv" and any(o in ("cls", "box") for o in n.output)]
    assert len(heads) == 2
    for n in heads:   # input and weights of the head do not pass through DequantizeLinear: the head stays float
        assert all(producer.get(i) is None or producer[i].op_type != "DequantizeLinear" for i in n.input[1:2])


def test_eval_onnx_scale_matches_torch_evaluate(tmp_path):
    from hae.export import export_onnx
    from hae.train import evaluate
    root, cache_dir = _tiny(tmp_path)
    res = _train(tmp_path, root, cache_dir)
    export_onnx(res["model"], tmp_path / "m.onnx", "lwir", (64, 96))
    cache, anns = Cache(cache_dir, "train"), load_annotations(root / "train_thermal.json", mode="coco3")
    a = evaluate(res["model"], cache, anns, [0, 1, 2, 3], mode="lwir", scale=2.0)
    b = eval_onnx(tmp_path / "m.onnx", cache, anns, [0, 1, 2, 3], mode="lwir", tnorm=True, ncls=3, scale=2.0)
    assert abs(a["AP50"] - b["AP50"]) < 0.05    # fp32 ONNX ~ PyTorch; differences only from operation order
    assert "per_class" in b


def test_calibration_uses_all_given_caches_at_inference_scale(tmp_path, monkeypatch):
    import hae.quantize as Q
    root, cache_dir = _tiny(tmp_path)
    _train(tmp_path, root, cache_dir)
    seen = {}
    real = Q.quantize_int8

    def spy(fp, out, calib):
        seen["shape"] = calib["t"][0].shape
        seen["n"] = len(calib["t"])
        return real(fp, out, calib)
    monkeypatch.setattr(Q, "quantize_int8", spy)
    c = Cache(cache_dir, "train")
    release_model(tmp_path / "run" / "last.pt", tmp_path / "rel", name="n_c", mode="lwir", calib_caches=[c, c], calib_n=6, tile_hw=(64, 96), scale=1.5)
    assert seen["n"] == 6 and seen["shape"] == (1, 1, 64, 96)    # 64×96 tile cut from a frame upscaled ×1.5 (64×96 -> 96×144)


def test_prf_table_counts_and_f1():
    gts = [np.array([[0, 0, 10, 10]], np.float32), np.zeros((0, 4), np.float32)]
    igs = [np.zeros((0, 4), np.float32)] * 2
    dets = [np.array([[0, 0, 10, 10, 0.9]], np.float32), np.array([[5, 5, 15, 15, 0.3]], np.float32)]
    t = prf_table(gts, igs, dets, (0.2, 0.5))
    assert t["0.2"]["tp"] == 1 and t["0.2"]["fp"] == 1 and abs(t["0.2"]["f1"] - 2 / 3) < 1e-6
    assert t["0.5"]["fp"] == 0 and t["0.5"]["f1"] == 1.0 and t["0.5"]["fp_per_frame"] == 0.0
