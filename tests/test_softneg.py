import torch

from hae.loss import HaeLoss


def _out(h=8, w=12, hot=(2, 3)):
    cls = torch.full((1, 1, h, w), -6.0)
    cls[0, 0, hot[0], hot[1]] = 4.0                 # a false positive in a cell next to a target
    box = torch.full((1, 4, h, w), 0.9)
    return {"cls": cls, "box": box}


BOX = [torch.tensor([[24.0, 8.0, 31.0, 15.0]])]     # centre (27.5, 11.5): cell (2,6); (2,5) is a neighbour
IGN = [torch.zeros(0, 4)]


def test_default_is_unchanged_and_decay_lowers_penalty_near_target():
    out = _out(hot=(2, 5))
    base = float(HaeLoss(batched=False)(out, BOX, IGN)[0])
    same = float(HaeLoss(batched=False, neg_decay=0.0)(out, BOX, IGN)[0])
    soft = float(HaeLoss(batched=False, neg_decay=1.0)(out, BOX, IGN)[0])
    assert base == same and soft < base


def test_decay_does_not_touch_far_cells_much_and_paths_agree():
    far = _out(hot=(7, 0))
    a = float(HaeLoss(batched=False)(far, BOX, IGN)[0])
    b = float(HaeLoss(batched=False, neg_decay=1.0)(far, BOX, IGN)[0])
    assert abs(a - b) / a < 0.01
    out = _out(hot=(2, 5))
    l1 = HaeLoss(batched=False, neg_decay=1.0)(out, BOX, IGN)[0]
    l2 = HaeLoss(batched=True, neg_decay=1.0)(out, BOX, IGN)[0]
    assert torch.allclose(l1, l2, atol=1e-5)
