"""hae loss: Quality Focal Loss (target = IoU in the channel of its class) + GIoU + NWD; ignore masks.

Two equivalent implementations (a test checks values and gradients): per-sample - faster on CPU with many targets,
batch-vectorized - for GPU, where the per-sample loop took 37 ms of a 62 ms step (T4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hae.boxes import box_iou_pairs, giou_pairs, nwd_pairs
from hae.model.decode import cell_centers, raw_to_boxes


def _t(x, dev):
    return torch.as_tensor(x, dtype=torch.float32, device=dev).reshape(-1, 4)


def _labels(labels, boxes, i):
    """Class labels of sample i; without labels - all 0 (single-class mode)."""
    n = len(np.asarray(boxes[i]).reshape(-1, 4))
    return np.zeros(n, np.int64) if labels is None else np.asarray(labels[i], np.int64).reshape(-1)


def _pad(lists, dev, labels=None):
    """list[np.ndarray (n_i,4)] -> (B,N,4), validity mask, (B,N) class labels (N >= 1)."""
    n = max([len(np.asarray(x).reshape(-1, 4)) for x in lists] + [1])
    arr = np.zeros((len(lists), n, 4), np.float32)
    valid = np.zeros((len(lists), n), bool)
    lab = np.zeros((len(lists), n), np.int64)
    for i, x in enumerate(lists):
        x = np.asarray(x, np.float32).reshape(-1, 4)
        arr[i, :len(x)], valid[i, :len(x)] = x, True
        if labels is not None:
            lab[i, :len(x)] = _labels(labels, lists, i)
    return torch.from_numpy(arr).to(dev), torch.from_numpy(valid).to(dev), torch.from_numpy(lab).to(dev)


class HaeLoss(nn.Module):
    def __init__(self, stride=4, nwd_c=12.8, cls_w=1.0, box_w=2.0, center_frac=0.25, min_half=0.75, batched=None,
                 class_weights=None, neg_decay: float = 0.0):
        super().__init__()
        self.neg_decay = float(neg_decay)   # >0: soft reduction of the negative penalty near a target (centre noise)
        self.class_weights = None if class_weights is None else tuple(float(v) for v in class_weights)   # weight of a class channel in QFL (rare classes up)
        self.stride, self.nwd_c, self.cls_w, self.box_w = stride, nwd_c, cls_w, box_w
        self.center_frac, self.min_half = center_frac, min_half
        self.last = None
        self.batched = batched   # None = auto: vectorized version on CUDA, per-sample on CPU

    def _regions(self, gt):
        """Half-sizes of the centre region (half-sides) for each target."""
        bw, bh = (gt[..., 2] - gt[..., 0]).clamp(min=1e-3), (gt[..., 3] - gt[..., 1]).clamp(min=1e-3)
        gx, gy = (gt[..., 0] + gt[..., 2]) / 2, (gt[..., 1] + gt[..., 3]) / 2
        hx = (bw * self.center_frac).clamp(min=self.min_half * self.stride)
        hy = (bh * self.center_frac).clamp(min=self.min_half * self.stride)
        return gx, gy, hx, hy

    def _neg_factor(self, gt, gvalid, cx, cy):
        """(1 - G)^beta for cells, G - max Gaussian over targets (sigma = max(side/4, stride)); gt: (...,G,4), gvalid: (...,G)."""
        gx, gy = (gt[..., 0] + gt[..., 2]) / 2, (gt[..., 1] + gt[..., 3]) / 2
        sx = ((gt[..., 2] - gt[..., 0]) / 4).clamp(min=self.stride)
        sy = ((gt[..., 3] - gt[..., 1]) / 4).clamp(min=self.stride)
        e = ((cx - gx[..., None, None]) / sx[..., None, None]) ** 2 + ((cy - gy[..., None, None]) / sy[..., None, None]) ** 2
        g = torch.exp(-0.5 * e).masked_fill(~gvalid[..., None, None], 0.0).amax(-3)
        return (1 - g).clamp(min=0).pow(self.neg_decay)

    def forward(self, out, boxes, ignore, labels=None):
        use_batched = self.batched if self.batched is not None else out["cls"].is_cuda
        return (self._forward_batched if use_batched else self._forward_loop)(out, boxes, ignore, labels)

    def _finish(self, cls, qt, weight, pb, gb, npos, pred_boxes):
        prob = cls.sigmoid()
        qfl = F.binary_cross_entropy_with_logits(cls, qt, reduction="none") * (prob - qt).abs().pow(2)
        if self.class_weights is not None:
            qfl = qfl * torch.tensor(self.class_weights, dtype=qfl.dtype, device=qfl.device).view(1, -1, 1, 1)
        cls_loss = (qfl * weight[:, None]).sum() / max(npos, 1)
        if npos:
            p, g = torch.cat(pb), torch.cat(gb)
            box_loss = ((1 - giou_pairs(p, g)) + (1 - nwd_pairs(p, g, self.nwd_c))).mean()
        else:
            box_loss = pred_boxes.sum() * 0.0
        loss = self.cls_w * cls_loss + self.box_w * box_loss
        self.last = {"qt": qt.detach(), "weight": weight.detach(), "npos": npos, "cls_loss": cls_loss}   # step state for distillation (distill.py); does not change the loss value
        return loss, {"cls": float(cls_loss.detach()), "box": float(box_loss.detach()), "npos": float(npos)}

    # ---- batch-vectorized (GPU) ----
    def _forward_batched(self, out, boxes, ignore, labels):
        cls = out["cls"]
        bsz, _, hg, wg = cls.shape
        dev = cls.device
        pred_boxes = raw_to_boxes(out["box"], self.stride)
        cy, cx = cell_centers(hg, wg, self.stride, dev)
        gt, gvalid, glab = _pad(boxes, dev, labels)
        ig, ivalid, _ = _pad(ignore, dev)
        gx, gy, hx, hy = (v[..., None, None] for v in self._regions(gt))
        dx, dy = (cx[None, None] - gx).abs(), (cy[None, None] - gy).abs()          # (B,G,H,W)
        inreg = (dx <= hx) & (dy <= hy) & gvalid[..., None, None]
        dist = ((dx / hx) ** 2 + (dy / hy) ** 2).masked_fill(~inreg, float("inf"))
        best, gi = dist.min(1)                                                    # (B,H,W)
        pos = torch.isfinite(best)
        gmatch = gt.gather(1, gi.flatten(1)[..., None].expand(-1, -1, 4)).view(bsz, hg, wg, 4)
        lmatch = glab.gather(1, gi.flatten(1)).view(bsz, hg, wg)
        inside = ((cx[None, None] >= ig[..., 0, None, None]) & (cx[None, None] <= ig[..., 2, None, None]) &
                  (cy[None, None] >= ig[..., 1, None, None]) & (cy[None, None] <= ig[..., 3, None, None]) &
                  ivalid[..., None, None]).any(1)
        weight = (~inside | pos).to(cls.dtype)          # a positive inside an ignore region stays positive
        if self.neg_decay > 0:
            weight = torch.where(pos, weight, weight * self._neg_factor(gt, gvalid, cx[None], cy[None]).to(cls.dtype))
        npos = int(pos.sum())
        qt = torch.zeros_like(cls)
        pb, gb = [], []
        if npos:
            bi, ys, xs = pos.nonzero(as_tuple=True)
            p, g = pred_boxes[pos], gmatch[pos]
            qt[bi, lmatch[pos], ys, xs] = box_iou_pairs(p.detach(), g)
            pb, gb = [p], [g]
        return self._finish(cls, qt, weight, pb, gb, npos, pred_boxes)

    # ---- per-sample (CPU) ----
    def _forward_loop(self, out, boxes, ignore, labels):
        cls = out["cls"]
        bsz, _, hg, wg = cls.shape
        dev = cls.device
        pred_boxes = raw_to_boxes(out["box"], self.stride)
        cy, cx = cell_centers(hg, wg, self.stride, dev)
        qt = torch.zeros_like(cls)
        weight = torch.ones(bsz, hg, wg, dtype=cls.dtype, device=dev)
        pb, gb = [], []
        for b in range(bsz):
            ig = _t(ignore[b], dev)
            if len(ig):
                inside = ((cx[None] >= ig[:, 0, None, None]) & (cx[None] <= ig[:, 2, None, None]) &
                          (cy[None] >= ig[:, 1, None, None]) & (cy[None] <= ig[:, 3, None, None]))
                weight[b][inside.any(0)] = 0.0
            gt = _t(boxes[b], dev)
            if len(gt):
                lab = torch.as_tensor(_labels(labels, boxes, b), device=dev)
                gx, gy, hx, hy = self._regions(gt)
                dx = (cx[None] - gx[:, None, None]).abs()
                dy = (cy[None] - gy[:, None, None]).abs()
                inreg = (dx <= hx[:, None, None]) & (dy <= hy[:, None, None])
                dist = ((dx / hx[:, None, None]) ** 2 + (dy / hy[:, None, None]) ** 2).masked_fill(~inreg, float("inf"))
                best, gi = dist.min(0)
                ys, xs = torch.isfinite(best).nonzero(as_tuple=True)
                if len(ys):
                    g, p = gt[gi[ys, xs]], pred_boxes[b, ys, xs]
                    qt[b, lab[gi[ys, xs]], ys, xs] = box_iou_pairs(p.detach(), g)
                    wneg = self._neg_factor(gt[None], torch.ones(1, len(gt), dtype=torch.bool, device=dev), cx, cy)[0] if self.neg_decay > 0 else None
                    if wneg is not None:
                        weight[b] = weight[b] * wneg
                    weight[b, ys, xs] = 1.0
                    pb.append(p)
                    gb.append(g)
        return self._finish(cls, qt, weight, pb, gb, sum(len(p) for p in pb), pred_boxes)


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, smooth: float = 0.8) -> torch.Tensor:
    """SoftIoU between sigmoid(logits) and a soft mask (HALO): robust to background/target imbalance."""
    p = logits.sigmoid()
    inter = (p * target).sum(dim=(1, 2, 3))
    union = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return (1.0 - (inter + smooth) / (union + smooth)).mean()
