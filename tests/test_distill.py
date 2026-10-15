import copy
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from hae.data.cache import build_cache
from hae.distill import DistillConfig, Distiller, OnnxTeacher, find_teacher, load_teacher
from hae.loss import HaeLoss
from hae.train import TrainConfig, build_model, train

DEV = torch.device("cpu")


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


def _teacher_ckpt(tmp_path, width=1.5, seed=1, **kw):
    torch.manual_seed(seed)
    tc = TrainConfig(width=width, **kw)
    m = build_model(tc)
    p = tmp_path / f"teacher_w{width}.pt"
    torch.save({"model": m.state_dict(), "cfg": asdict(tc), "epoch": 1}, p)
    return p


def _cfg(tmp_path, root, cache_dir, name="run", **kw):
    base = dict(cache_dir=str(cache_dir), ann_root=str(root), out_dir=str(tmp_path / name), epochs=2, crops_per_epoch=32, batch=8,
                lr=3e-3, warmup_steps=2, eval_every=99, threads=2, crop_hw=(32, 64), mode_probs=(0, 0, 1), tb=False, quick_val=0, ckpt_every=1)
    base.update(kw)
    return TrainConfig(**base)


# ---------- config ----------
def test_neutral_config_is_disabled_and_validates():
    assert not DistillConfig.from_dict(None).enabled
    assert not DistillConfig.from_dict({"teacher": "x.pt", "alpha": 0.0, "feat": False}).enabled
    assert DistillConfig.from_dict({"alpha": 0.5}).enabled and DistillConfig.from_dict({"feat": True}).enabled
    assert DistillConfig.from_dict({"alpha": 0.5}).box_a == 0.5
    with pytest.raises(ValueError):
        DistillConfig.from_dict({"alpha": 0.5, "bogus": 1})
    with pytest.raises(ValueError):
        DistillConfig.from_dict({"comb": "sum"})
    with pytest.raises(ValueError):
        DistillConfig.from_dict({"alpha": 1.5})


def test_enabled_without_teacher_fails_loudly(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path)
    with pytest.raises(ValueError, match="teacher"):
        train(_cfg(tmp_path, root, cache_dir, distill={"alpha": 0.5}))


def test_find_teacher_resolves_relative_to_bases(tmp_path):
    d = tmp_path / "base" / "hae-teachers" / "sub"
    d.mkdir(parents=True)
    (d / "hae-s-ir.pt").write_bytes(b"x")
    assert find_teacher(str(d / "hae-s-ir.pt")) == d / "hae-s-ir.pt"
    assert find_teacher("hae-teachers/sub/hae-s-ir.pt", bases=[str(tmp_path / "base")]) == d / "hae-s-ir.pt"
    assert find_teacher("hae-teachers/hae-s-ir.pt".replace("hae-teachers/", "sub/"), bases=[str(tmp_path / "base")]) == d / "hae-s-ir.pt"
    with pytest.raises(FileNotFoundError):
        find_teacher("nope/x.pt", bases=[str(tmp_path / "base")])


# ---------- teacher ----------
def test_load_teacher_is_frozen_eval_and_matches_checkpoint(tmp_path):
    p = _teacher_ckpt(tmp_path)
    t = load_teacher(str(p), DEV)
    assert not t.training and all(not q.requires_grad for q in t.parameters())
    ck = torch.load(p, weights_only=False)
    assert all(torch.equal(a, b) for a, b in zip(t.state_dict().values(), ck["model"].values()))


def _batch(h=32, w=64, bsz=4):
    g = torch.Generator().manual_seed(0)
    t = torch.rand(bsz, 1, h, w, generator=g)
    boxes = [np.array([[10, 8, 20, 18]], np.float32) for _ in range(bsz)]
    return {"t": t}, boxes, [np.zeros((0, 4), np.float32)] * bsz


def _student_step(student, dist, loss_fn, kw, boxes, ign, opt):
    out = student(**kw)
    loss, lg = loss_fn({k: v.float() for k, v in out.items()}, boxes, ign, None)
    tout = dist.teach(kw)
    extra, logs = dist.loss({k: v.float() for k, v in out.items()}, tout, loss_fn, boxes)
    total = loss + extra
    opt.zero_grad()
    total.backward()
    opt.step()
    return float(total.detach()), logs


@pytest.mark.parametrize("ratio", [1.0, 4 / 3])
def test_loss_decreases_teacher_untouched(tmp_path, ratio):
    torch.manual_seed(0)
    student = build_model(TrainConfig(width=0.67)).train()
    teacher = load_teacher(str(_teacher_ckpt(tmp_path)), DEV)
    before = copy.deepcopy(teacher.state_dict())
    cfg = DistillConfig(teacher="x", alpha=0.7, tau=1.0, small_boost=1.0, feat=True, feat_levels=2, teacher_scale_ratio=ratio)
    dist = Distiller(cfg, student, DEV, False, teacher=teacher)
    loss_fn = HaeLoss(stride=4)
    opt = torch.optim.AdamW(list(student.parameters()) + list(dist.parameters()), lr=3e-3)
    kw, boxes, ign = _batch()
    first, logs = _student_step(student, dist, loss_fn, kw, boxes, ign, opt)
    for _ in range(25):
        last, logs = _student_step(student, dist, loss_fn, kw, boxes, ign, opt)
    assert np.isfinite(last) and last < first
    assert {"kd_cls", "kd_feat"} <= set(logs)
    assert not teacher.training and all(a is not None for a in before.values())
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())   # parameters and BN buffers unchanged
    assert all(q.grad is None for q in teacher.parameters())


def test_soft_target_pulls_student_towards_teacher_heatmap():
    """No hard labels: a cell with high teacher confidence raises the student logit, background does not grow."""
    class Fix(torch.nn.Module):
        cfg = type("C", (), {"stride": 4, "ctx": False, "tnorm": False, "ncls": 1, "neck_c": 8})()

        def forward(self, rgb=None, t=None):
            z = torch.full((1, 1, 8, 16), -5.0)
            z[0, 0, 4, 8] = 3.0
            return {"cls": z, "box": torch.zeros(1, 4, 8, 16)}
    student = type("S", (), {"cfg": Fix.cfg})()
    dist = Distiller(DistillConfig(teacher="x", alpha=1.0, box_alpha=0.0), student, DEV, False, teacher=Fix())
    loss_fn = HaeLoss(stride=4)
    z = torch.full((1, 1, 8, 16), -3.0, requires_grad=True)
    box = torch.zeros(1, 4, 8, 16)
    opt = torch.optim.SGD([z], lr=1.0)
    for _ in range(60):
        loss_fn({"cls": z, "box": box}, [np.zeros((0, 4), np.float32)], [np.zeros((0, 4), np.float32)])
        kd, _ = dist.loss({"cls": z, "box": box}, dist.teach({"t": torch.zeros(1, 1, 32, 64)}), loss_fn, [np.zeros((0, 4), np.float32)])
        opt.zero_grad()
        kd.backward()
        opt.step()
    assert float(z[0, 0, 4, 8].detach()) > 0.0 > float(z[0, 0, 0, 0].detach())


def test_cross_scale_pooling_maps_peak_and_box_to_student_grid():
    class Fix(torch.nn.Module):
        cfg = type("C", (), {"stride": 4, "ctx": False, "tnorm": False, "ncls": 1, "neck_c": 8})()

        def forward(self, rgb=None, t=None):
            z = torch.full((1, 1, 16, 32), -6.0)
            z[0, 0, 9, 13] = 4.0        # teacher cell (x=13,y=9) -> centre (54, 38) teacher px
            box = torch.full((1, 4, 16, 32), -6.0)
            box[0, :, 9, 13] = 1.0      # l=t=r=b=softplus(1)*4 ~ 5.25 teacher px
            return {"cls": z, "box": box}
    student = type("S", (), {"cfg": Fix.cfg})()
    dist = Distiller(DistillConfig(teacher="x", alpha=0.5, teacher_scale_ratio=2.0), student, DEV, False, teacher=Fix())
    tout = dist.teach({"t": torch.zeros(1, 1, 32, 64)})
    assert tout["rh"] == tout["rw"] == 2.0   # teacher input 64x128 -> grid 16x32
    z, b = dist._to_student_grid(tout, 8, 16)
    assert z.shape == (1, 1, 8, 16) and int(z.flatten().argmax()) == 4 * 16 + 6   # (13,9) -> (6,4)
    exp = torch.tensor([54.0 - 5.2531, 38.0 - 5.2531, 54.0 + 5.2531, 38.0 + 5.2531]) / 2
    assert torch.allclose(b[0, 4, 6], exp, atol=1e-3)


def test_small_boost_only_near_small_teacher_peaks():
    class Fix(torch.nn.Module):
        cfg = type("C", (), {"stride": 4, "ctx": False, "tnorm": False, "ncls": 1, "neck_c": 8})()

        def forward(self, rgb=None, t=None):
            z = torch.full((1, 1, 8, 16), -6.0)
            z[0, 0, 2, 3] = 4.0          # small target: box ~ 2*softplus(-1)*4 ~ 3 px
            z[0, 0, 5, 12] = 4.0         # large target
            box = torch.full((1, 4, 8, 16), -1.0)
            box[0, :, 5, 12] = 3.0       # softplus(3)*4 ~ 12.5 px per side -> 25 px
            return {"cls": z, "box": box}
    student = type("S", (), {"cfg": Fix.cfg})()
    dist = Distiller(DistillConfig(teacher="x", alpha=0.5, small_boost=2.0, small_px=16.0), student, DEV, False, teacher=Fix())
    loss_fn = HaeLoss(stride=4)
    z = torch.zeros(1, 1, 8, 16)
    box = torch.zeros(1, 4, 8, 16)
    loss_fn({"cls": z, "box": box}, [np.zeros((0, 4), np.float32)], [np.zeros((0, 4), np.float32)])
    _, logs = dist.loss({"cls": z, "box": box}, dist.teach({"t": torch.zeros(1, 1, 32, 64)}), loss_fn, [np.zeros((0, 4), np.float32)])
    assert logs["kd_small_cells"] == 9.0   # only 3x3 around the small peak


def test_feature_distillation_gives_adapter_and_student_gradients(tmp_path):
    student = build_model(TrainConfig(width=0.67)).train()
    teacher = load_teacher(str(_teacher_ckpt(tmp_path)), DEV)
    dist = Distiller(DistillConfig(teacher="x", alpha=0.0, feat=True, feat_levels=2), student, DEV, False, teacher=teacher)
    assert dist.cfg.enabled and len(list(dist.parameters())) == 4   # 2 levels x (weight, bias)
    kw, boxes, ign = _batch()
    loss_fn = HaeLoss(stride=4)
    out = student(**kw)
    loss_fn({k: v.float() for k, v in out.items()}, boxes, ign, None)
    extra, logs = dist.loss(out, dist.teach(kw), loss_fn, boxes)
    extra.backward()
    assert logs["kd_feat"] > 0 and all(p.grad is not None and p.grad.abs().sum() > 0 for p in dist.parameters())
    assert student.fuse4[0].weight.grad is not None


def test_mismatched_ncls_or_tnorm_raises(tmp_path):
    student = build_model(TrainConfig(width=0.67, classes=1))
    t3 = load_teacher(str(_teacher_ckpt(tmp_path, classes=3)), DEV)
    with pytest.raises(ValueError, match="class"):
        Distiller(DistillConfig(teacher="x", alpha=0.5), student, DEV, False, teacher=t3)
    tn = load_teacher(str(_teacher_ckpt(tmp_path, tnorm=True)), DEV)
    with pytest.raises(ValueError, match="tnorm"):
        Distiller(DistillConfig(teacher="x", alpha=0.5), student, DEV, False, teacher=tn)


def test_onnx_teacher_logits_only(tmp_path):
    pytest.importorskip("onnxruntime")
    from hae.export import export_onnx
    m = build_model(TrainConfig(width=1.5)).eval()
    p = tmp_path / "t.onnx"
    export_onnx(m, p, mode="lwir", hw=(32, 64))
    teacher = OnnxTeacher(p)
    student = build_model(TrainConfig(width=0.67)).train()
    dist = Distiller(DistillConfig(teacher="x", alpha=0.5, teacher_scale_ratio=4 / 3), student, DEV, False, teacher=teacher)
    kw, boxes, ign = _batch()
    loss_fn = HaeLoss(stride=4)
    out = student(**kw)
    loss_fn({k: v.float() for k, v in out.items()}, boxes, ign, None)
    extra, logs = dist.loss(out, dist.teach(kw), loss_fn, boxes)
    extra.backward()
    assert np.isfinite(float(extra.detach())) and "kd_cls" in logs
    with pytest.raises(ValueError, match="ONNX"):
        Distiller(DistillConfig(teacher="x", feat=True), student, DEV, False, teacher=teacher)


# ---------- integration into train() ----------
def test_alpha_zero_is_identical_to_no_distill(tmp_path):
    root, cache_dir = _tiny_dataset(tmp_path)
    t = _teacher_ckpt(tmp_path)
    a = train(_cfg(tmp_path, root, cache_dir, "a"))
    b = train(_cfg(tmp_path, root, cache_dir, "b", distill={"teacher": str(t), "alpha": 0.0, "tau": 2.0, "feat": False, "teacher_scale_ratio": 1.5}))
    assert [h["loss"] for h in a["history"]] == [h["loss"] for h in b["history"]]
    assert all(torch.equal(x, y) for x, y in zip(a["model"].state_dict().values(), b["model"].state_dict().values()))
    assert not any(k.startswith("kd_") for k in b["history"][0])


@pytest.mark.parametrize("extra", [dict(alpha=0.6), dict(alpha=0.5, small_boost=1.0, teacher_scale_ratio=4 / 3, comb="mix"), dict(alpha=0.3, feat=True, feat_levels=2)])
def test_train_with_distill_runs_logs_and_resumes(tmp_path, extra):
    root, cache_dir = _tiny_dataset(tmp_path)
    t = _teacher_ckpt(tmp_path)
    cfg = _cfg(tmp_path, root, cache_dir, epochs=1, distill={"teacher": str(t), **extra}, ema_decay=0.9)
    r1 = train(cfg)["history"]
    assert np.isfinite(r1[0]["loss"]) and "kd_cls" in r1[0]
    ck = torch.load(tmp_path / "run" / "resume.pt", weights_only=False)
    assert ("kd" in ck) == bool(extra.get("feat"))
    cfg2 = _cfg(tmp_path, root, cache_dir, epochs=2, resume=True, distill={"teacher": str(t), **extra}, ema_decay=0.9)
    r2 = train(cfg2)["history"]
    assert len(r2) == 2 and r2[0] == r1[0] and np.isfinite(r2[1]["loss"])
    # the student checkpoint contains neither teacher nor adapters: release/inference unchanged
    best = torch.load(tmp_path / "run" / "last.pt", weights_only=False)
    assert not any("adapter" in k or "teacher" in k for k in best["model"])
    ref = build_model(TrainConfig()).state_dict()
    assert set(best["model"]) == set(ref)
