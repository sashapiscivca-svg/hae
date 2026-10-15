
import numpy as np
import torch

from hae.loss import HaeLoss


def _case(seed, b=6, hg=16, wg=20, gmax=12, imax=4):
    rng = np.random.default_rng(seed)
    cls = torch.tensor(rng.normal(-3, 1.5, (b, 1, hg, wg)), dtype=torch.float32, requires_grad=True)
    box = torch.tensor(rng.normal(0.5, 0.6, (b, 4, hg, wg)), dtype=torch.float32, requires_grad=True)
    boxes, ign = [], []
    for _ in range(b):
        n = int(rng.integers(0, gmax))   # including empty frames
        xy = rng.uniform(0, 60, (n, 2)); wh = rng.uniform(4, 22, (n, 2))
        boxes.append(np.concatenate([xy, xy + wh], 1).astype(np.float32).reshape(-1, 4))
        m = int(rng.integers(0, imax))
        xy = rng.uniform(0, 60, (m, 2)); wh = rng.uniform(8, 40, (m, 2))
        ign.append(np.concatenate([xy, xy + wh], 1).astype(np.float32).reshape(-1, 4))
    return {"cls": cls, "box": box}, boxes, ign


def test_vectorised_loss_matches_reference_values_and_gradients():
    for seed in range(8):
        out, boxes, ign = _case(seed)
        l_new, log_new = HaeLoss(batched=True)(out, boxes, ign)
        g_new = torch.autograd.grad(l_new, [out["cls"], out["box"]])
        out2 = {k: v.detach().clone().requires_grad_(True) for k, v in out.items()}
        l_ref, log_ref = HaeLoss(batched=False)(out2, boxes, ign)
        g_ref = torch.autograd.grad(l_ref, [out2["cls"], out2["box"]])
        assert torch.allclose(l_new, l_ref, atol=1e-5, rtol=1e-5), (seed, float(l_new), float(l_ref))
        assert log_new["npos"] == log_ref["npos"]
        for a, b in zip(g_new, g_ref):
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_all_empty_batch_and_overlapping_gts():
    out, _, _ = _case(0, b=3)
    e = [np.zeros((0, 4), np.float32)] * 3
    assert float(HaeLoss(batched=True)(out, e, e)[0]) == float(HaeLoss(batched=False)(out, e, e)[0])
    same = [np.array([[10, 10, 20, 20], [11, 10, 21, 20]], np.float32)] * 3     # overlap: nearest target
    assert torch.allclose(HaeLoss(batched=True)(out, same, e)[0], HaeLoss(batched=False)(out, same, e)[0], atol=1e-5)


def test_auto_mode_uses_loop_on_cpu_and_both_paths_agree_there():
    out, boxes, ign = _case(3)
    auto = HaeLoss()(out, boxes, ign)[0]
    assert torch.equal(auto, HaeLoss(batched=False)(out, boxes, ign)[0])


def _mc_case(seed):
    out, boxes, ign = _case(seed)
    k = 3
    rng = np.random.default_rng(seed + 100)
    out = {"cls": torch.tensor(rng.normal(-3, 1.5, (6, k, 16, 20)), dtype=torch.float32, requires_grad=True), "box": out["box"]}
    labels = [rng.integers(0, k, len(b)).astype(np.int64) for b in boxes]
    return out, boxes, ign, labels


def test_multiclass_loss_paths_agree_on_values_and_gradients():
    for seed in range(6):
        out, boxes, ign, labels = _mc_case(seed)
        l_b, log_b = HaeLoss(batched=True)(out, boxes, ign, labels)
        g_b = torch.autograd.grad(l_b, [out["cls"], out["box"]])
        out2 = {k_: v.detach().clone().requires_grad_(True) for k_, v in out.items()}
        l_l, log_l = HaeLoss(batched=False)(out2, boxes, ign, labels)
        g_l = torch.autograd.grad(l_l, [out2["cls"], out2["box"]])
        assert torch.allclose(l_b, l_l, atol=1e-5, rtol=1e-5) and log_b["npos"] == log_l["npos"]
        for a, b in zip(g_b, g_l):
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_multiclass_target_goes_to_the_labelled_channel():
    for batched in (False, True):
        gt = [np.array([[28, 20, 38, 30]], np.float32)]
        none = [np.zeros((0, 4), np.float32)]
        def run(good_ch, lab):
            cls = torch.full((1, 3, 16, 20), -6.0)
            cls[0, good_ch, 5:6, 7:8] = 3.0      # target centre ~ cell (5,7)
            out = {"cls": cls, "box": torch.full((1, 4, 16, 20), 0.9)}
            return float(HaeLoss(batched=batched)(out, gt, none, [np.array([lab], np.int64)])[1]["cls"])
        assert run(2, 2) < run(0, 2) and run(1, 1) < run(2, 1)


def test_class_weights_scale_each_class_channel_and_paths_agree():
    out, boxes, ign, labels = _mc_case(2)
    w = (1.0, 3.0, 2.0)
    for batched in (False, True):
        base = HaeLoss(batched=batched)(out, boxes, ign, labels)[0]
        weighted = HaeLoss(batched=batched, class_weights=w)(out, boxes, ign, labels)[0]
        assert float(weighted) > float(base)                      # weights >= 1 increase the loss
    a = HaeLoss(batched=False, class_weights=w)(out, boxes, ign, labels)[0]
    b = HaeLoss(batched=True, class_weights=w)(out, boxes, ign, labels)[0]
    assert torch.allclose(a, b, atol=1e-5)
    ones = HaeLoss(batched=False, class_weights=(1.0, 1.0, 1.0))(out, boxes, ign, labels)[0]
    assert torch.allclose(ones, HaeLoss(batched=False)(out, boxes, ign, labels)[0])
