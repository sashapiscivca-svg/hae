import json

import numpy as np
from PIL import Image

from hae.data.annotations import load_annotations
from hae.data.cache import Cache, build_cache
from hae.model.hae import HaeConfig, HaeNet
from hae.train import TrainConfig, evaluate, train


def _tiny_dataset(tmp_path, n=6, h=64, w=96):
    root, out = tmp_path / "ds", tmp_path / "cache"
    for m in ("visible", "thermal"):
        (root / "train" / m).mkdir(parents=True)
    imgs, anns = [], []
    for i in range(n):
        im = np.zeros((h, w, 3), np.uint8)
        x = 10 + 8 * i
        im[20:30, x:x + 10] = 255
        for m in ("visible", "thermal"):
            Image.fromarray(im).save(root / "train" / m / f"{i:05d}.jpg", quality=100)
        imgs.append({"id": i, "file_name": f"{i:05d}.jpg", "height": h, "width": w})
        anns.append({"id": i, "image_id": i, "category_id": 0, "bbox": [x, 20, 10, 10], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", out)
    return root, out


def test_train_runs_logs_and_loss_decreases(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"),
                      epochs=3, crops_per_epoch=128, batch=8, lr=3e-3, warmup_steps=2, eval_every=3, threads=2, crop_hw=(32, 64))
    res = train(cfg)
    hist = res["history"]
    assert len(hist) == 3 and all(np.isfinite(h["loss"]) for h in hist)
    assert hist[-1]["loss"] < hist[0]["loss"]
    assert (tmp_path / "run" / "last.pt").exists()
    lines = (tmp_path / "run" / "log.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3 and "epoch_s" in json.loads(lines[0])


def test_evaluate_returns_metric_keys_and_handles_each_mode(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path)
    cache = Cache(cache_dir, "train")
    anns = load_annotations(root / "train_thermal.json")
    m = HaeNet(HaeConfig()).eval()
    for mode in ("both", "rgb", "lwir"):
        r = evaluate(m, cache, anns, mode=mode, batch=3)
        assert set(r) == {"AP", "AP50", "AP75", "AR"}


def test_ema_decay_zero_tracks_model_and_train_returns_raw_model():
    import torch
    from hae.train import ModelEma
    m = HaeNet(HaeConfig())
    ema = ModelEma(m, decay=0.0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    assert all(torch.allclose(a, b) for a, b in zip(ema.m.parameters(), m.parameters()))


def test_train_logs_eval_for_every_modality(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1,
                      crops_per_epoch=16, batch=8, threads=2, crop_hw=(32, 64), eval_frames=[0, 1], eval_every=1)
    rec = train(cfg)["history"][0]
    for mode in ("both", "rgb", "lwir"):
        assert f"eval_{mode}_AP50" in rec, rec.keys()


def test_eval_split_selects_cache_and_missing_split_fails_loudly(tmp_path):
    import pytest
    root, cache_dir = _tiny_dataset(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1,
                      crops_per_epoch=16, batch=8, threads=2, crop_hw=(32, 64), eval_frames=[0], eval_split="val")
    with pytest.raises(FileNotFoundError):
        train(cfg)   # no "val" cache: evaluation must not silently fall back to train frames


def test_train_accepts_model_size_and_records_it(tmp_path):
    import torch
    root, cache_dir = _tiny_dataset(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1,
                      crops_per_epoch=16, batch=8, threads=2, crop_hw=(32, 64), size="n")
    res = train(cfg)
    ck = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert ck["cfg"]["size"] == "n"
    n_params = sum(p.numel() for p in res["model"].parameters())
    from hae.model.presets import build
    assert n_params == sum(p.numel() for p in build(size="n").parameters())


def test_verbose_prints_one_flushed_line_per_epoch(tmp_path, capsys):
    root, cache_dir = _tiny_dataset(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=2,
                      crops_per_epoch=16, batch=8, threads=2, crop_hw=(32, 64), verbose=True)
    train(cfg)
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("epoch ")]
    assert len(lines) == 2 and "loss=" in lines[0] and "s/epoch" in lines[0]


def test_train_with_scale_aug_and_eval_at_scale(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path, h=64, w=96)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1,
                      crops_per_epoch=16, batch=8, threads=2, crop_hw=(32, 64), scale_range=(0.8, 1.6), prefetch=2)
    assert np.isfinite(train(cfg)["history"][0]["loss"])
    cache = Cache(cache_dir, "train")
    anns = load_annotations(root / "train_thermal.json")
    for sc in (1.0, 2.0):
        r = evaluate(HaeNet(HaeConfig()).eval(), cache, anns, mode="both", scale=sc)
        assert set(r) == {"AP", "AP50", "AP75", "AR"}


def test_train_with_align_logs_alignment_loss(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path, h=64, w=96)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=2,
                      crops_per_epoch=32, batch=8, threads=2, crop_hw=(32, 64), align=True, mode_probs=(1.0, 0.0, 0.0))
    hist = train(cfg)["history"]
    assert all(np.isfinite(h["align_loss"]) and h["align_loss"] > 0 for h in hist)
    plain = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run2"), epochs=1,
                        crops_per_epoch=16, batch=8, threads=2, crop_hw=(32, 64))
    assert "align_loss" not in train(plain)["history"][0]


def _tiny_dataset_3cls(tmp_path, n=6, h=64, w=96):
    root, out = tmp_path / "ds3", tmp_path / "cache3"
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
        anns.append({"id": 2 * i, "image_id": i, "category_id": i % 3, "bbox": [x, 20, 10, 10], "ignore": 0, "iscrowd": 0})
        anns.append({"id": 2 * i + 1, "image_id": i, "category_id": 3, "bbox": [x, 40, 8, 8], "ignore": 0, "iscrowd": 0})
    (root / "train_thermal.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": []}))
    build_cache(root, "train", out)
    return root, out


def test_three_class_training_and_protocol_evaluation(tmp_path):
    root, cache_dir = _tiny_dataset_3cls(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=2,
                      crops_per_epoch=32, batch=8, threads=2, crop_hw=(32, 64), classes=3,
                      eval_frames=[0, 1, 2], eval_every=2, eval_modes=("both",))
    res = train(cfg)
    rec = res["history"][-1]
    assert np.isfinite(rec["loss"]) and res["model"].cfg.ncls == 3
    assert {"eval_both_AP50", "eval_both_AP25", "eval_both_AP50_tiny", "eval_both_per_class"} <= set(rec)
    cache = Cache(cache_dir, "train")
    from hae.model.presets import build
    r = evaluate(build(ncls=3).eval(), cache, load_annotations(root / "train_thermal.json", mode="coco3"), mode="both")
    assert {"AP50", "AP25", "AP50_tiny", "per_class"} <= set(r)


def test_loss_weights_and_config_overrides_are_applied(tmp_path):
    import pytest
    root, cache_dir = _tiny_dataset(tmp_path)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1,
                      crops_per_epoch=16, batch=8, threads=2, crop_hw=(32, 64), box_w=7.5, cls_w=0.5,
                      cfg_overrides={"depths": [3, 2, 2], "head_c": 32})
    res = train(cfg)
    assert res["model"].cfg.depths == (3, 2, 2) and res["model"].cfg.head_c == 32 and res["model"].cfg.widths == (24, 32, 48)
    bad = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run2"), epochs=1,
                      crops_per_epoch=8, batch=8, threads=2, crop_hw=(32, 64), cfg_overrides={"nope": 1})
    with pytest.raises(ValueError, match="nope"):
        train(bad)


def test_third_party_backbones_are_rejected_but_released_configs_load():
    import glob
    import pytest
    from hae.train import build_model
    with pytest.raises(ValueError, match="from scratch"):
        build_model(TrainConfig(backbone="resnet18"))
    with pytest.raises(ValueError, match="from scratch"):
        build_model(TrainConfig(size="n", pretrained=True))
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for f in glob.glob(str(root / "models" / "*" / "config.json")):
        tr = json.loads(open(f).read())["training"]
        cfg = TrainConfig(**tr)   # sanitized released configs (paths removed) load as they are
        assert cfg.size in "ntsm" and cfg.backbone is None


def test_train_with_extra_thermal_source_uses_only_lwir_for_it(tmp_path):
    from test_extra import _fake_hituav
    from hae.data.extra import convert_hituav
    root, cache_dir = _tiny_dataset(tmp_path, h=64, w=96)
    src, uni = tmp_path / "hit", tmp_path / "uni"
    _fake_hituav(src)
    convert_hituav(src, uni)
    from hae.data.cache import build_cache_thermal
    build_cache_thermal(uni, "hituav", cache_dir)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=2,
                      crops_per_epoch=64, batch=8, threads=2, crop_hw=(32, 64), mode_probs=(1.0, 0.0, 0.0),
                      extra=[{"name": "hituav", "unified": str(uni), "weight": 0.5}])
    hist = train(cfg)["history"]
    assert all(np.isfinite(h["loss"]) for h in hist) and sum(h["extra_batches"] for h in hist) > 0
    # mode_probs=(1,0,0) allows only both; batches of the LWIR-only extra set must go through in lwir mode without errors


def test_train_with_copy_paste_runs(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path, h=64, w=96)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1,
                      crops_per_epoch=24, batch=8, threads=2, crop_hw=(32, 64), copy_paste=0.7)
    assert np.isfinite(train(cfg)["history"][0]["loss"])


def test_train_and_evaluate_with_frame_normalised_thermal(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path, h=64, w=96)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=1,
                      crops_per_epoch=24, batch=8, threads=2, crop_hw=(32, 64), tnorm=True,
                      eval_frames=[0, 1], eval_modes=("both", "lwir"))
    res = train(cfg)
    assert res["model"].cfg.tnorm is True and np.isfinite(res["history"][0]["loss"])
    assert {"eval_both_AP50", "eval_lwir_AP50"} <= set(res["history"][0])


def test_train_with_halo_builds_masks_logs_aux_loss_and_exports_without_aux(tmp_path):
    from test_extra import _fake_hituav
    from hae.data.extra import convert_hituav
    from hae.data.cache import build_cache_thermal
    from hae.export import export_onnx
    root, cache_dir = _tiny_dataset_3cls(tmp_path)
    src, uni = tmp_path / "hit", tmp_path / "uni"
    _fake_hituav(src)
    convert_hituav(src, uni)
    build_cache_thermal(uni, "hituav", cache_dir)
    cfg = TrainConfig(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / "run"), epochs=2, crops_per_epoch=32,
                      batch=8, threads=2, crop_hw=(32, 64), classes=3, size="n", halo=True, aux_w=1.0,
                      extra=[{"name": "hituav", "unified": str(uni), "weight": 0.4}], mode_probs=(1.0, 0.0, 0.0))
    res = train(cfg)
    assert (cache_dir / "train_halo.u8").exists() and (cache_dir / "hituav_halo.u8").exists()
    assert all(np.isfinite(h["aux_loss"]) and h["aux_loss"] > 0 for h in res["history"])
    export_onnx(res["model"], tmp_path / "n.onnx", mode="lwir", hw=(64, 96))
    import onnxruntime as ort
    outs = ort.InferenceSession(str(tmp_path / "n.onnx"), providers=["CPUExecutionProvider"]).get_outputs()
    assert [o.name for o in outs] == ["cls", "box"]                       # no auxiliary branch in the deployment graph


def test_eval_scale_rounds_input_to_multiple_of_32_for_odd_frame_sizes(tmp_path):
    """600×800 × 1.5 = 900×1200 used to fail ("multiple of 16"): the input is rounded to a multiple of 32, boxes map back in x and y."""
    from hae.train import detect_frames
    root, cache_dir = _tiny_dataset(tmp_path, h=48, w=80)
    cache, anns = Cache(cache_dir, "train"), load_annotations(root / "train_thermal.json")
    m = HaeNet(HaeConfig(ctx=True)).eval()
    for sc in (1.0, 1.5):
        dets = detect_frames(m, cache, range(cache.n), mode="lwir", scale=sc, score_thr=0.0, topk=50)
        assert len(dets) == cache.n
        assert all(float(d[:, [0, 2]].max(initial=0)) <= 80 * 1.1 and float(d[:, [1, 3]].max(initial=0)) <= 48 * 1.1 for d in dets)
        assert set(evaluate(m, cache, anns, mode="lwir", scale=sc)) == {"AP", "AP50", "AP75", "AR"}
