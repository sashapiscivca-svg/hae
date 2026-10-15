"""Decoding HaeNet outputs into boxes without NMS."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def cell_centers(hg: int, wg: int, stride: int, device=None):
    ys = (torch.arange(hg, device=device, dtype=torch.float32) + 0.5) * stride
    xs = (torch.arange(wg, device=device, dtype=torch.float32) + 0.5) * stride
    return torch.meshgrid(ys, xs, indexing="ij")


def raw_to_boxes(box_raw: torch.Tensor, stride: int) -> torch.Tensor:
    """(B,4,Hg,Wg) raw -> (B,Hg,Wg,4) xyxy; distances l,t,r,b = softplus(raw)*stride."""
    _, _, hg, wg = box_raw.shape
    cy, cx = cell_centers(hg, wg, stride, box_raw.device)
    l, t, r, b = (F.softplus(box_raw) * stride).unbind(1)
    return torch.stack([cx - l, cy - t, cx + r, cy + b], -1)


@torch.no_grad()
def decode(out: dict, stride: int = 4, topk: int = 100, score_thr: float = 0.05) -> list:
    """No NMS: 3x3 peaks per class. K=1 -> (n,5) [x1,y1,x2,y2,score]; K>1 -> (n,6) with class."""
    score = out["cls"].sigmoid()
    bsz, k, hg, wg = score.shape
    peaks = score * (score == F.max_pool2d(score, 3, 1, 1)).float()
    boxes = raw_to_boxes(out["box"], stride)
    res = []
    for b in range(bsz):
        s, idx = peaks[b].flatten().topk(min(topk, k * hg * wg))
        keep = s >= score_thr
        s, idx = s[keep], idx[keep]
        bx = boxes[b].reshape(-1, 4)[idx % (hg * wg)]
        cols = [bx, s[:, None]] + ([(idx // (hg * wg)).float()[:, None]] if k > 1 else [])
        res.append(torch.cat(cols, 1).cpu().numpy().astype(np.float32))
    return res
