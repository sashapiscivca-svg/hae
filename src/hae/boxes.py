"""IoU, GIoU and NWD (Normalized Wasserstein Distance) for pairs of xyxy boxes.

NWD: a box is the Gaussian N((cx,cy), diag(w^2/4, h^2/4)); the 2-Wasserstein distance has a closed form.
Useful for targets < 16 px, where IoU drops to 0 even for a 1-2 px shift.
"""
from __future__ import annotations

import torch

EPS = 1e-7


def _wh(b: torch.Tensor):
    return (b[:, 2] - b[:, 0]).clamp(min=0), (b[:, 3] - b[:, 1]).clamp(min=0)


def _inter_union(a: torch.Tensor, b: torch.Tensor):
    ix = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy
    wa, ha = _wh(a)
    wb, hb = _wh(b)
    return inter, wa * ha + wb * hb - inter


def box_iou_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter, union = _inter_union(a, b)
    return inter / (union + EPS)


def giou_pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter, union = _inter_union(a, b)
    iou = inter / (union + EPS)
    cw = torch.maximum(a[:, 2], b[:, 2]) - torch.minimum(a[:, 0], b[:, 0])
    ch = torch.maximum(a[:, 3], b[:, 3]) - torch.minimum(a[:, 1], b[:, 1])
    carea = cw.clamp(min=0) * ch.clamp(min=0)
    return iou - (carea - union) / (carea + EPS)


def nwd_pairs(a: torch.Tensor, b: torch.Tensor, c: float = 12.8) -> torch.Tensor:
    acx, acy = (a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2
    bcx, bcy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    wa, ha = _wh(a)
    wb, hb = _wh(b)
    w2 = (acx - bcx) ** 2 + (acy - bcy) ** 2 + ((wa - wb) / 2) ** 2 + ((ha - hb) / 2) ** 2
    return torch.exp(-torch.sqrt(w2 + 1e-9) / c)
