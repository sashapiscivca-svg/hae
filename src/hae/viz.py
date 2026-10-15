"""Detection visualisation: draws boxes and confidences on a frame; for TensorBoard, the CLI and examples."""
from __future__ import annotations

import cv2
import numpy as np

PRED, GT = (255, 64, 64), (64, 220, 64)   # RGB: red = prediction, green = ground truth


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    elif a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.moveaxis(a, 0, -1)
        a = np.repeat(a, 3, -1) if a.shape[-1] == 1 else a
    return np.ascontiguousarray(a)


def draw_detections(image: np.ndarray, dets: np.ndarray, gts: np.ndarray | None = None, thr: float = 0.3, show_score: bool = True) -> np.ndarray:
    """image: (H,W), (H,W,3) or (C,H,W), uint8 or float; dets: (n,5|6) [x1,y1,x2,y2,score(,cls)]; gts: (m,4). Returns a new (H,W,3) uint8."""
    out = _to_rgb_u8(image).copy()
    for b in np.asarray(gts if gts is not None else np.zeros((0, 4)), np.float32).reshape(-1, 4):
        cv2.rectangle(out, (int(round(b[0])), int(round(b[1]))), (int(round(b[2])), int(round(b[3]))), GT, 1)
    d = np.asarray(dets, np.float32)
    d = d.reshape(len(d), -1) if d.size else np.zeros((0, 5), np.float32)
    for r in d[d[:, 4] >= thr]:
        p1, p2 = (int(round(r[0])), int(round(r[1]))), (int(round(r[2])), int(round(r[3])))
        cv2.rectangle(out, p1, p2, PRED, 1)
        if show_score:
            cv2.putText(out, f"{r[4]:.2f}", (p1[0], max(p1[1] - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, PRED, 1, cv2.LINE_AA)
    return out
