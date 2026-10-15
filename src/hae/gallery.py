"""Ready-to-view images: a "prediction vs. ground truth" grid, an error gallery (missed people and false alarms) and a single dashboard.
Everything is written to PNG automatically during training, so TensorBoard is not required."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from hae.viz import _to_rgb_u8, draw_detections


def split_errors(gts, dets, conf: float, margin: float = 2.0, cluster_px: float = 8.0):
    """Splits errors with the same logic as recall_center: [(frame, box)] of missed people and [(frame, box, confidence)] of false alarms."""
    from hae.recall import _center_nms
    fn, fp = [], []
    for k, (g, d) in enumerate(zip(gts, dets)):
        g = np.asarray(g, np.float32).reshape(-1, 4)
        d = np.asarray(d, np.float32)
        d = d.reshape(len(d), -1) if d.size else np.zeros((0, 5), np.float32)
        d = _center_nms(d[d[:, 4] >= conf] if len(d) else d, cluster_px)
        used = np.zeros(len(g), bool)
        for r in d:
            cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
            hit = next((j for j, b in enumerate(g) if not used[j] and b[0] - margin <= cx <= b[2] + margin and b[1] - margin <= cy <= b[3] + margin), -1)
            if hit >= 0:
                used[hit] = True
            else:
                fp.append((k, r[:4].copy(), float(r[4])))
        fn += [(k, g[j].copy()) for j in range(len(g)) if not used[j]]
    return fn, fp


def pred_grid(images, gts, dets, conf: float, cols: int = 3, width: int = 480) -> np.ndarray:
    """Frame grid: ground truth in green, predictions in red (threshold conf); each frame is resized to width `width`."""
    tiles = []
    for im, g, d in zip(images, gts, dets):
        t = draw_detections(im, d, gts=g, thr=conf, show_score=False)
        h, w = t.shape[:2]
        tiles.append(cv2.resize(t, (width, max(int(round(h * width / w)), 1)), interpolation=cv2.INTER_AREA))
    hh = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * hh, cols * width, 3), np.uint8)
    for k, t in enumerate(tiles):
        canvas[(k // cols) * hh:(k // cols) * hh + t.shape[0], (k % cols) * width:(k % cols + 1) * width] = t
    return canvas


def _crop(img: np.ndarray, box, margin: int = 20, zoom: int = 4) -> np.ndarray:
    a = _to_rgb_u8(img)
    h, w = a.shape[:2]
    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    x0, y0 = max(cx - margin, 0), max(cy - margin, 0)
    c = a[y0:min(cy + margin, h), x0:min(cx + margin, w)].copy()
    cv2.rectangle(c, (int(box[0]) - x0, int(box[1]) - y0), (int(box[2]) - x0, int(box[3]) - y0), (255, 64, 64), 1)
    return cv2.resize(c, (c.shape[1] * zoom, c.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)


def error_gallery(images, gts, dets, conf: float, path: str | Path, n: int = 12) -> Path:
    """Two rows of zoomed crops: missed people (FN) and the most confident false alarms (FP), showing at a glance what the model confuses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fn, fp = split_errors(gts, dets, conf)
    fp = sorted(fp, key=lambda x: -x[2])[:n]
    fn = fn[:n]
    cols = min(max(len(fn), len(fp), 1), 6)
    fig, axes = plt.subplots(2, cols, figsize=(2.7 * cols, 6.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for row, (items, title) in enumerate(((fn, f"missed people (FN): {len(fn)}"), (fp, f"false alarms (FP, top by confidence): {len(fp)}"))):
        for j, it in enumerate(items[:cols]):
            axes[row, j].imshow(_crop(images[it[0]], it[1], margin=24, zoom=6))
            if row == 1:   # confidence below the crop so it does not overlap the row title
                axes[row, j].text(0.5, -0.03, f"conf {it[2]:.2f}", transform=axes[row, j].transAxes, ha="center", va="top", fontsize=8)
        axes[row, 0].set_title(title, fontsize=10, fontweight="bold", loc="left", pad=12)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.03, hspace=0.28, wspace=0.08)
    p = Path(path)
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def dashboard(history: list, curves: dict | None, grid: np.ndarray | None, path: str | Path, title: str = "") -> Path:
    """One sheet: loss, AP/AR per epoch, P/R/F1 curves and the prediction vs. ground-truth grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hae.plots import draw_curves
    ep = [r["epoch"] for r in history]
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.7])
    a = fig.add_subplot(gs[0, 0])
    a.plot(ep, [r["loss"] for r in history], lw=1.8)
    a.set_title("training loss", fontsize=10)
    a.grid(alpha=0.3)
    b = fig.add_subplot(gs[0, 1])
    for key, lab in (("qval", "quick validation"), ("eval", "full evaluation")):
        ks = sorted({k for r in history for k in r if k.startswith(key + "_") and k.endswith("_AP50")})
        for k in ks:
            xs = [r["epoch"] for r in history if k in r]
            b.plot(xs, [r[k] for r in history if k in r], marker="o", ms=3, lw=1.6, label=f"{lab} AP50")
    b.set_title("validation AP50", fontsize=10)
    b.grid(alpha=0.3)
    if b.get_legend_handles_labels()[0]:
        b.legend(fontsize=7)
    if curves:
        draw_curves(fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]), curves)
    d = fig.add_subplot(gs[1, :])
    d.axis("off")
    if grid is not None:
        d.imshow(grid)
        d.set_title("validation: ground truth (green) and predictions (red)", fontsize=10)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = Path(path)
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p
