import torch

from hae.boxes import box_iou_pairs, giou_pairs, nwd_pairs


def t(*rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_identical_boxes():
    a = t([0, 0, 10, 10])
    assert torch.allclose(box_iou_pairs(a, a), torch.ones(1), atol=1e-5)
    assert torch.allclose(giou_pairs(a, a), torch.ones(1), atol=1e-5)
    assert torch.allclose(nwd_pairs(a, a), torch.ones(1), atol=1e-3)


def test_disjoint_giou_negative_iou_zero():
    a, b = t([0, 0, 10, 10]), t([20, 0, 30, 10])
    assert float(box_iou_pairs(a, b)) == 0.0
    assert float(giou_pairs(a, b)) < 0.0


def test_nwd_smooth_where_iou_is_zero():
    a = t([0, 0, 10, 10])
    near, far = t([12, 0, 22, 10]), t([40, 0, 50, 10])
    assert float(box_iou_pairs(a, near)) == 0.0 and float(box_iou_pairs(a, far)) == 0.0
    assert float(nwd_pairs(a, near)) > float(nwd_pairs(a, far)) > 0.0


def test_degenerate_boxes_no_nan_and_grad():
    a = t([5, 5, 5, 5], [0, 0, 0, 10])
    b = t([5, 5, 5, 5], [0, 0, 4, 4]).clone().requires_grad_(True)
    for fn in (box_iou_pairs, giou_pairs, nwd_pairs):
        out = fn(a, b)
        assert torch.isfinite(out).all()
    nwd_pairs(a, b).sum().backward()
    assert torch.isfinite(b.grad).all()
