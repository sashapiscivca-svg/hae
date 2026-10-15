"""SAHI tiling for large drone frames: overlapping tiles, shift to frame coordinates, duplicate merging.

A lightweight numpy implementation (no heavy dependencies, runs next to onnxruntime on a Pi). Idea: Akyon et al., SAHI (2022).
The network is fully convolutional, so a tile can match the training crop (256x320) and a 4K frame is processed as a batch of tiles.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def _axis(n: int, t: int, overlap: float) -> list[int]:
    if n <= t:
        return [0]
    step = max(int(t * (1.0 - overlap)), 1)
    org = list(range(0, n - t + 1, step))
    if org[-1] != n - t:
        org.append(n - t)          # the last tile is snapped to the edge: the frame is fully covered
    return org


def tile_origins(h: int, w: int, th: int, tw: int, overlap: float = 0.2) -> list[tuple[int, int]]:
    return [(y, x) for y in _axis(h, th, overlap) for x in _axis(w, tw, overlap)]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    iw = (np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0])).clip(0)
    ih = (np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1])).clip(0)
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def merge_detections(dets: np.ndarray, iou_thr: float = 0.5) -> np.ndarray:
    """Greedy merging of duplicates in overlap zones: by descending score, per class; the best box is kept."""
    dets = np.asarray(dets, np.float32)
    if len(dets) == 0:
        return dets.reshape(0, dets.shape[1] if dets.ndim == 2 else 6)
    cls = dets[:, 5] if dets.shape[1] > 5 else np.zeros(len(dets), np.float32)
    order, keep = np.argsort(-dets[:, 4]), []
    alive = np.ones(len(dets), bool)
    for i in order:
        if not alive[i]:
            continue
        keep.append(i)
        same = alive & (cls == cls[i])
        same[i] = False
        idx = np.nonzero(same)[0]
        if len(idx):
            alive[idx[_iou(dets[i, :4], dets[idx, :4]) > iou_thr]] = False
    return dets[keep]


def sliced_predict(infer: Callable, rgb: np.ndarray | None, t: np.ndarray | None, tile_hw=(256, 320), overlap: float = 0.2,
                   iou_thr: float = 0.5) -> np.ndarray:
    """infer(rgb_tile, t_tile, (y0, x0)) -> (n,5|6) in tile coordinates. rgb: (3,H,W) or None; t: (1,H,W) or None.
    Apply the per-FRAME thermal normalisation (tnorm) before calling, so that all tiles share the same scale."""
    ref = rgb if rgb is not None else t
    h, w = ref.shape[-2:]
    th, tw = tile_hw
    out = []
    for y0, x0 in tile_origins(h, w, th, tw, overlap):
        def tile(a, y0=y0, x0=x0):
            if a is None:
                return None
            crop = a[:, y0:y0 + th, x0:x0 + tw]
            pad = np.zeros((a.shape[0], th, tw), a.dtype)
            pad[:, :crop.shape[1], :crop.shape[2]] = crop
            return pad
        d = np.asarray(infer(tile(rgb), tile(t), (y0, x0)), np.float32)
        if d.size:
            d = d.reshape(len(d), -1)          # (n,5) for one class or (n,6) with class
            d = d.copy()
            d[:, [0, 2]] += x0
            d[:, [1, 3]] += y0
            out.append(d)
    if not out:
        return np.zeros((0, 6), np.float32)
    d = np.concatenate(out)
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, w)
    d[:, [1, 3]] = d[:, [1, 3]].clip(0, h)
    d = d[((d[:, 2] - d[:, 0]) > 0.5) & ((d[:, 3] - d[:, 1]) > 0.5)]
    return merge_detections(d, iou_thr)
