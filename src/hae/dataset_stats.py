"""Label statistics and batch previews (labels / train_batch images) at the start of training, to inspect data and augmentations at a glance."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from hae.viz import draw_detections


def label_stats(anns, names) -> dict:
    boxes = [a.boxes for a in anns if len(a.boxes)]
    wh = np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32)
    size = np.sqrt(np.maximum((wh[:, 2] - wh[:, 0]) * (wh[:, 3] - wh[:, 1]), 0)) if len(wh) else np.zeros(0)
    labs = np.concatenate([a.labels for a in anns if len(a.boxes)]) if boxes else np.zeros(0, int)
    n = max(len(wh), 1)
    return {"frames": len(anns), "empty_frames": sum(1 for a in anns if not len(a.boxes)), "instances": len(wh),
            "per_class": {names[k] if k < len(names) else str(k): int((labs == k).sum()) for k in sorted(set(labs.tolist()))},
            "size_p10": float(np.percentile(size, 10)) if len(size) else 0.0, "size_p50": float(np.median(size)) if len(size) else 0.0,
            "size_p90": float(np.percentile(size, 90)) if len(size) else 0.0, "small_lt10px": float((size < 10).sum() / n),
            "per_frame_mean": len(wh) / max(len(anns), 1), "ignore_regions": int(sum(len(a.ignore) for a in anns))}


def plot_labels(anns, path: str | Path, names=("person",)) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wh = np.concatenate([a.boxes for a in anns if len(a.boxes)]) if any(len(a.boxes) for a in anns) else np.zeros((0, 4), np.float32)
    h, w = anns[0].height, anns[0].width
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    size = np.sqrt((wh[:, 2] - wh[:, 0]) * (wh[:, 3] - wh[:, 1])) if len(wh) else np.zeros(0)
    ax[0].hist(size, bins=40, color="#3b7dd8")
    ax[0].axvline(10, color="r", ls="--", lw=1)
    ax[0].set_title(f"person size sqrt(area), px (p50 {np.median(size) if len(size) else 0:.1f})", fontsize=9)
    ax[1].hist2d((wh[:, 0] + wh[:, 2]) / 2 / w, (wh[:, 1] + wh[:, 3]) / 2 / h, bins=24, cmap="viridis") if len(wh) else None
    ax[1].set_title("box centre locations in the frame", fontsize=9)
    ax[1].invert_yaxis()
    cnt = np.array([len(a.boxes) for a in anns])
    ax[2].hist(cnt, bins=range(0, int(cnt.max()) + 2) if len(cnt) else 1, color="#3b7dd8")
    ax[2].set_title("people per frame", fontsize=9)
    fig.tight_layout()
    p = Path(path)
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


def save_batch_preview(batch, path: str | Path, mode: str = "lwir", n: int = 8, cols: int = 4) -> Path:
    """Grid of the first n crops of a batch AFTER augmentation with labels (green): shows at once whether augmentation broke the data."""
    from PIL import Image
    imgs = []
    for k in range(min(n, len(batch.boxes))):
        if mode == "rgb":
            a = batch.rgb[k].permute(1, 2, 0).numpy()
        else:
            t = batch.t[k, 0].numpy()
            a = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
        imgs.append(draw_detections(a, np.zeros((0, 5), np.float32), gts=batch.boxes[k]))
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for k, im in enumerate(imgs):
        canvas[(k // cols) * h:(k // cols + 1) * h, (k % cols) * w:(k % cols + 1) * w] = im
    p = Path(path)
    Image.fromarray(canvas).save(p)
    return p
