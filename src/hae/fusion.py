"""Weighted box fusion for TTA. Idea from Weighted Boxes Fusion (arXiv 1910.13302), own implementation: IoU clustering,
confidence-weighted mean coordinates, confidence = mean * min(members, T) / T, where T is the number of fused runs (a lone detection is weakened)."""
from __future__ import annotations

import numpy as np


def _iou_one(b: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = (np.minimum(b[2], boxes[:, 2]) - np.maximum(b[0], boxes[:, 0])).clip(0)
    iy = (np.minimum(b[3], boxes[:, 3]) - np.maximum(b[1], boxes[:, 1])).clip(0)
    inter = ix * iy
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def fuse_wbf(dets_list: list, iou_thr: float = 0.4) -> np.ndarray:
    """dets_list: list of (n,6) arrays [x1,y1,x2,y2,score,class] from different runs -> one (m,6) array."""
    t = len(dets_list)
    allb = [np.asarray(d, np.float32).reshape(len(d), -1) for d in dets_list if len(d)]
    if not allb:
        return np.zeros((0, 6), np.float32)
    d = np.concatenate([x if x.shape[1] == 6 else np.concatenate([x, np.zeros((len(x), 1), np.float32)], 1) for x in allb])
    d = d[np.argsort(-d[:, 4])]
    clusters, fused = [], []
    for row in d:
        hit = -1
        cand = [i for i, c in enumerate(clusters) if fused[i][5] == row[5]]
        if cand:
            ious = _iou_one(row[:4], np.stack([fused[i][:4] for i in cand]))
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                hit = cand[j]
        if hit < 0:
            clusters.append([row])
            fused.append(row.copy())
        else:
            clusters[hit].append(row)
            m = np.stack(clusters[hit])
            w = m[:, 4:5]
            fused[hit][:4] = (m[:, :4] * w).sum(0) / w.sum()
    out = []
    for c, f in zip(clusters, fused):
        m = np.stack(c)
        f = f.copy()
        f[4] = m[:, 4].mean() * min(len(m), t) / t
        out.append(f)
    out = np.stack(out)
    return out[np.argsort(-out[:, 4])]
