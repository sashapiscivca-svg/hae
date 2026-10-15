import numpy as np
import torch

from hae.loss import HaeLoss
from hae.model.hae import HaeConfig, HaeNet


def _out(b=1, hg=16, wg=20, requires_grad=True):
    cls = torch.full((b, 1, hg, wg), -4.0, requires_grad=requires_grad)
    box = torch.full((b, 4, hg, wg), 0.9, requires_grad=requires_grad)
    return {"cls": cls, "box": box}


E = np.zeros((0, 4), np.float32)


def test_empty_targets_finite_and_backward():
    out = _out()
    loss, logs = HaeLoss()(out, [E], [E])
    assert torch.isfinite(loss) and logs["npos"] == 0
    loss.backward()
    assert torch.isfinite(out["cls"].grad).all()


def test_one_target_gives_positives_and_box_gradient():
    out = _out()
    gt = [np.array([[28, 20, 38, 30]], np.float32)]
    loss, logs = HaeLoss()(out, gt, [E])
    assert logs["npos"] >= 1 and logs["box"] > 0
    loss.backward()
    assert out["box"].grad.abs().sum() > 0 and out["cls"].grad.abs().sum() > 0


def test_ignore_region_removes_negative_loss():
    out = _out(requires_grad=False)
    whole = [np.array([[0, 0, 80, 64]], np.float32)]
    loss, logs = HaeLoss()(out, [E], whole)
    assert logs["cls"] == 0.0 and float(loss) == 0.0


def test_ignore_does_not_cancel_positive_cells():
    out = _out(requires_grad=False)
    gt = [np.array([[28, 20, 38, 30]], np.float32)]
    whole = [np.array([[0, 0, 80, 64]], np.float32)]
    _, logs = HaeLoss()(out, gt, whole)
    assert logs["npos"] >= 1 and logs["cls"] > 0


def test_degenerate_gt_box_is_finite():
    out = _out()
    loss, _ = HaeLoss()(out, [np.array([[30, 30, 30, 30]], np.float32)], [E])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(out["box"].grad).all()


def test_batch_with_mixed_frames_and_real_model():
    m = HaeNet(HaeConfig())
    rgb, t = torch.rand(3, 3, 64, 96), torch.rand(3, 1, 64, 96)
    gts = [np.array([[10, 10, 20, 20], [40, 30, 52, 44]], np.float32), E, np.array([[60, 20, 70, 32]], np.float32)]
    loss, logs = HaeLoss()(m(rgb=rgb, t=t), gts, [E, E, np.array([[0, 0, 30, 30]], np.float32)])
    loss.backward()
    assert torch.isfinite(loss) and logs["npos"] >= 3
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
