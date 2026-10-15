"""Decoding of hae outputs in pure numpy, for deployment without PyTorch (e.g. Raspberry Pi Zero 2 W + onnxruntime).

Matches hae.model.decode.decode: 3x3 peaks per class, top-K, l,t,r,b = softplus(raw) * stride from the cell centre.
"""
from __future__ import annotations

import numpy as np


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _peak_mask(score: np.ndarray) -> np.ndarray:
    """(K,H,W) -> mask of 3x3 local maxima (equal neighbours are both kept, as with torch max_pool)."""
    p = np.pad(score, ((0, 0), (1, 1), (1, 1)), constant_values=-np.inf)
    h, w = score.shape[1:]
    mx = p[:, 0:h, 0:w].copy()
    for dy in range(3):
        for dx in range(3):
            np.maximum(mx, p[:, dy:dy + h, dx:dx + w], out=mx)
    return score == mx


def decode_np(cls: np.ndarray, box: np.ndarray, stride: int = 4, topk: int = 100, score_thr: float = 0.05) -> list:
    """cls (B,K,H,W) logits, box (B,4,H,W) raw. K=1 -> (n,5) [x1,y1,x2,y2,score]; K>1 -> (n,6) with class."""
    out = []
    b_, k, hg, wg = cls.shape
    ys = (np.arange(hg, dtype=np.float32) + 0.5) * stride
    xs = (np.arange(wg, dtype=np.float32) + 0.5) * stride
    cy, cx = np.meshgrid(ys, xs, indexing="ij")
    for b in range(b_):
        score = _sigmoid(cls[b].astype(np.float32))
        peaks = np.where(_peak_mask(score), score, 0.0).reshape(-1)
        idx = np.argsort(-peaks, kind="stable")[:min(topk, peaks.size)]
        idx = idx[peaks[idx] >= score_thr]
        cell = idx % (hg * wg)
        d = _softplus(box[b].astype(np.float32).reshape(4, -1)[:, cell]) * stride       # softplus only on selected cells (the full map costs ~37 ms per call on a Pi)
        cxs, cys = cx.reshape(-1)[cell], cy.reshape(-1)[cell]
        cols = [cxs - d[0], cys - d[1], cxs + d[2], cys + d[3], peaks[idx]] + ([(idx // (hg * wg)).astype(np.float32)] if k > 1 else [])
        out.append(np.stack(cols, 1).astype(np.float32))
    return out
