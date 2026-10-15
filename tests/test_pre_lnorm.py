"""Local contrast normalization as input preprocessing (pre_lnorm), polarity inversion (t_invert), lnorm channel window lnorm_k."""
import json

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from hae.data.annotations import load_annotations
from hae.data.cache import Cache, build_cache, ensure_lnorm_cache, lnorm_cache, lnorm_k_of
from hae.data.sampler import CropSampler, load_frames
from hae.infer import default_metadata, local_contrast_u8, preprocess_thermal
from hae.model.hae import HaeConfig, local_norm
from hae.model.presets import build


def _ref(g, k):
    f = g.astype(np.float32)
    m = cv2.blur(f, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - m * m, 1))
    return np.clip(128 + 32 * (f - m) / (sd + 4), 0, 255).astype(np.uint8)


def test_local_contrast_constant_dtype_shape_and_formula():
    c = np.full((64, 96), 200, np.uint8)
    out = local_contrast_u8(c, 61)
    assert out.dtype == np.uint8 and out.shape == (64, 96) and (out == 128).all()
    g = np.random.default_rng(0).integers(0, 255, (70, 90), np.uint8)
    assert np.array_equal(local_contrast_u8(g, 61), _ref(g, 61))
    assert np.array_equal(local_contrast_u8(np.stack([g] * 3, -1), 15), _ref(g, 15))   # (H,W,3) -> first channel


def test_local_contrast_lifts_blob_in_bright_surrounding():
    rng = np.random.default_rng(1)
    g = np.clip(rng.normal(215, 3, (128, 160)), 0, 255).astype(np.uint8)   # "heated ground"
    g[60:66, 70:74] = 235                                                    # a person barely warmer than the background
    raw_contrast = float(g[60:66, 70:74].mean()) - float(g.mean())
    out = local_contrast_u8(g, 61)
    lifted = float(out[60:66, 70:74].mean()) - float(out[:40, :40].mean())
    assert lifted > 3 * raw_contrast and out[60:66, 70:74].mean() > 200


def _tiny(tmp_path, n=4, h=64, w=96):
    root = tmp_path / "ds"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    rng = np.random.default_rng(0)
    for i in range(n):
        t = rng.integers(150, 200, (h, w), np.uint8)
        x = 10 + 10 * i
        t[20:30, x:x + 10] = 240
        Image.fromarray(np.stack([t] * 3, -1)).save(root / "train" / "visible" / f"{i}.png")
        Image.fromarray(t).save(root / "train" / "thermal" / f"{i}.png")
        imgs.append({"id": i, "file_name": f"{i}.png", "height": h, "width": w})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [x, 20, 10, 10], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", tmp_path / "c")
    return root, tmp_path / "c"


def test_lnorm_cache_transforms_thermal_links_rgb_and_recomputes_stats(tmp_path):
    root, cdir = _tiny(tmp_path)
    raw = Cache(cdir, "train")
    _ = raw.tstats                                                  # creates train_tstats.npy in the raw cache
    dst = ensure_lnorm_cache(cdir, 61)
    assert dst == tmp_path / "c_lnorm61" and lnorm_k_of(dst) == 61 and lnorm_k_of(cdir) == 0
    assert (dst / "train_rgb.u8").is_symlink() and not (dst / "train_tstats.npy").exists()
    c = Cache(dst, "train")
    assert c.pre_lnorm == 61 and raw.pre_lnorm == 0
    for i in range(c.n):
        assert np.array_equal(c.t[i], local_contrast_u8(raw.t[i], 61))
    m, s = c.tstats
    assert np.allclose(m[0], np.asarray(c.t[0], np.float32).mean(), atol=1e-3)   # statistics come from converted frames
    assert ensure_lnorm_cache(dst, 61) == dst                        # already converted - left untouched
    with pytest.raises(ValueError):
        lnorm_cache(dst, tmp_path / "x", 61)                         # double normalization is forbidden
    # evaluation: raw cache + pre_lnorm (on the fly) == converted cache
    for tn in (False, True):
        _, a = load_frames(raw, [0, 2], t_norm=tn, pre_lnorm=61)
        _, b = load_frames(c, [0, 2], t_norm=tn, pre_lnorm=61)
        assert torch.allclose(a, b, atol=1e-5)
    with pytest.raises(ValueError):
        load_frames(c, [0], pre_lnorm=0)                             # a model without pre_lnorm on a converted cache


def test_t_invert_flips_polarity_and_tnorm_sign(tmp_path):
    root, cdir = _tiny(tmp_path)
    cache, anns = Cache(cdir, "train"), load_annotations(root / "train_thermal.json")
    for tn in (False, True):
        a = CropSampler(cache, anns, crop_hw=(32, 48), seed=4, t_norm=tn).sample_batch(1)
        b = CropSampler(cache, anns, crop_hw=(32, 48), seed=4, t_norm=tn, t_invert=1.0).sample_batch(1)
        ref = -a.t if tn else 1.0 - a.t
        assert torch.allclose(b.t, ref, atol=1e-5) and np.allclose(a.boxes[0], b.boxes[0])
    x = CropSampler(cache, anns, crop_hw=(32, 48), seed=4).sample_batch(4)
    y = CropSampler(cache, anns, crop_hw=(32, 48), seed=4, t_invert=0.0).sample_batch(4)
    assert torch.equal(x.t, y.t)                                     # off = previous behaviour (rng untouched)


def test_metadata_and_detector_without_key_are_unchanged(tmp_path):
    pytest.importorskip("onnxruntime")
    from hae.detector import Detector
    from hae.export import export_onnx
    from hae.infer import read_metadata
    assert "pre_lnorm" not in default_metadata("lwir", 1, True, 4)
    assert default_metadata("lwir", 1, True, 4, 0) == default_metadata("lwir", 1, True, 4)
    g = np.random.default_rng(2).integers(100, 160, (64, 96), np.uint8)
    g[20:28, 30:36] = 200
    old = build("n", tnorm=True).eval()
    p_old = tmp_path / "old.onnx"
    export_onnx(old, p_old, "lwir", (64, 96))
    assert "pre_lnorm" not in read_metadata(p_old)
    d_old = Detector(p_old, conf=0.0)
    assert d_old.pre_lnorm == 0
    x = preprocess_thermal(g, True)
    ref = d_old.sess.run(None, {d_old.input: x})[0]
    import onnxruntime as ort
    s = ort.InferenceSession(str(p_old), providers=["CPUExecutionProvider"])
    assert np.array_equal(ref, s.run(None, {"t": x})[0])
    base = d_old.detect(g)
    assert np.array_equal(base, Detector(p_old, conf=0.0).detect(g))
    # model with pre_lnorm: key in metadata, Detector applies local_contrast_u8 on the full frame
    new = build("n", tnorm=True, overrides={"pre_lnorm": 61}).eval()
    p_new = tmp_path / "new.onnx"
    export_onnx(new, p_new, "lwir", (64, 96))
    assert read_metadata(p_new)["pre_lnorm"] == 61
    d_new = Detector(p_new, conf=0.0)
    assert d_new.pre_lnorm == 61
    same_weights_raw = Detector(p_old, conf=0.0)
    new.load_state_dict(old.state_dict())
    export_onnx(new, p_new, "lwir", (64, 96))
    d_new = Detector(p_new, conf=0.0)
    assert np.array_equal(d_new.detect(g), same_weights_raw.detect(local_contrast_u8(g, 61)))
    assert np.array_equal(d_new.detect(g, scale=1.5), same_weights_raw.detect(local_contrast_u8(g, 61), scale=1.5))


def test_lnorm_k_default_is_unchanged_and_configurable(tmp_path):
    assert HaeConfig().lnorm_k == 31 and HaeConfig().pre_lnorm == 0
    assert HaeConfig(lnorm=True, lnorm_k=61).scaled(1.67).lnorm_k == 61
    torch.manual_seed(0)
    m31 = build("n", overrides={"lnorm": True}).eval()
    m61 = build("n", overrides={"lnorm": True, "lnorm_k": 61}).eval()
    m61.load_state_dict(m31.state_dict())                           # same weights: only the window differs
    x = torch.rand(1, 1, 128, 160)
    with torch.no_grad():
        assert not torch.allclose(m31(t=x)["cls"], m61(t=x)["cls"])
        assert torch.equal(local_norm(x), local_norm(x, 31))         # old behaviour: k=31
    from hae.export import export_onnx
    assert export_onnx(m61, tmp_path / "k61.onnx", mode="lwir", hw=(128, 160))["bytes"] > 0


def test_train_smoke_with_pre_lnorm_t_invert_and_lnorm2ch(tmp_path):
    from hae.train import TrainConfig, train
    root, cdir = _tiny(tmp_path)
    for name, kw in (("pre", {"pre_lnorm": 61, "t_invert": 0.3, "tnorm": True}),
                     ("ch2", {"t_invert": 0.3, "tnorm": True, "cfg_overrides": {"lnorm": True, "lnorm_k": 61}})):
        cfg = TrainConfig(cache_dir=str(cdir), ann_root=str(root), out_dir=str(tmp_path / name), epochs=1, crops_per_epoch=16, batch=8,
                          warmup_steps=1, eval_every=1, eval_frames=[0, 1], eval_modes=("lwir",), mode_probs=(0, 0, 1), threads=1,
                          crop_hw=(32, 64), quick_val=2, tb=False, **kw)
        res = train(cfg)
        assert np.isfinite(res["history"][0]["loss"]) and "eval_lwir_AP50" in res["history"][0]
        ck = torch.load(tmp_path / name / "last.pt", weights_only=False)
        if name == "pre":
            assert res["model"].cfg.pre_lnorm == 61 and ck["cfg"]["pre_lnorm"] == 61 and lnorm_k_of(tmp_path / "c_lnorm61") == 61
        else:
            assert res["model"].cfg.lnorm_k == 61 and res["model"].cfg.pre_lnorm == 0
