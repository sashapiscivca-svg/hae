"""HALO-like soft masks built from boxes for thermal targets.

Idea from the paper on physically grounded soft labels for IR targets (Hotspot-Anchored Label Optimization, arXiv 2607.17148):
an IR target is a compact hot blob, and a hard box (especially a noisy one, sigma~2 px in RGBTDronePerson) describes it poorly. So the mask
is anchored to the warmest point inside the box (signal/noise relative to the local background), not the geometric centre, and has a Gaussian
profile. Written from scratch from the description; used ONLY as an auxiliary training target (zero runtime cost).
"""
from __future__ import annotations

import cv2
import numpy as np


def halo_mask(thermal: np.ndarray, boxes: np.ndarray, ctx_ratio: float = 1.5, tau: float = 3.0, gamma: float = 1.5) -> np.ndarray:
    """thermal (H,W) uint8/float; boxes (N,4) xyxy -> (H,W) float32 in [0,1] (max over boxes)."""
    img = cv2.GaussianBlur(thermal.astype(np.float32), (0, 0), 1.0)
    h, w = img.shape
    out = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in np.asarray(boxes, np.float32).reshape(-1, 4):
        bx1, by1, bx2, by2 = max(int(np.floor(x1)), 0), max(int(np.floor(y1)), 0), min(int(np.ceil(x2)), w), min(int(np.ceil(y2)), h)
        if bx2 - bx1 < 2 or by2 - by1 < 2:
            continue
        cx, cy, bw, bh = (bx1 + bx2) / 2, (by1 + by2) / 2, bx2 - bx1, by2 - by1
        cx1, cx2 = max(int(cx - ctx_ratio * bw / 2), 0), min(int(np.ceil(cx + ctx_ratio * bw / 2)), w)
        cy1, cy2 = max(int(cy - ctx_ratio * bh / 2), 0), min(int(np.ceil(cy + ctx_ratio * bh / 2)), h)
        ctx = img[cy1:cy2, cx1:cx2]
        s = (img[by1:by2, bx1:bx2] - ctx.mean()) / (ctx.std() + 1e-4)          # local "signal/noise"
        smax = float(s.max())
        yy, xx = np.mgrid[by1:by2, bx1:bx2]
        if smax <= 0.0:                                                         # no hot blob: box centre
            ay, ax, sg = cy, cx, max(0.25 * min(bw, bh), 1.0)
            wgt = np.full_like(s, 0.5)
        else:
            iy, ix = np.unravel_index(int(np.argmax(s)), s.shape)
            ay, ax = by1 + iy, bx1 + ix
            half = s >= 0.5 * smax
            r2 = (yy - ay) ** 2 + (xx - ax) ** 2
            sp = np.where(half, np.clip(s, 0, None), 0.0)
            sg = max(gamma * float(np.sqrt((sp * r2).sum() / (sp.sum() + 1e-9))), 1.0)
            wgt = 1.0 / (1.0 + np.exp(-s / tau))
            wgt = (wgt - wgt.min()) / (wgt.max() - wgt.min() + 1e-9)
        g = np.exp(-((yy - ay) ** 2 + (xx - ax) ** 2) / (2 * sg * sg))
        out[by1:by2, bx1:bx2] = np.maximum(out[by1:by2, bx1:bx2], (wgt * g).astype(np.float32))
    return out
