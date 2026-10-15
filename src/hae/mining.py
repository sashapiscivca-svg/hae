"""Hard-negative mining: points where the model confidently "sees" a person that is not in the annotation.

Idea: Kellenberger et al. (arXiv 1806.11368): after warm-up, fine-tuning with emphasis on false positives reduces them several-fold.
Here the model runs over part of the training frames, false peaks are stored as (frame, cx, cy, score), and CropSampler
centres a crop on such a point with probability hn_frac. No external models - only the current EMA model.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from hae.data.sampler import load_frames
from hae.model.decode import decode


def mine_from_detections(dets, anns, frames, score_thr: float = 0.2, match_margin: float = 4.0, per_frame: int = 10) -> np.ndarray:
    """dets[j] - (n,6) [x1,y1,x2,y2,score(,cls)] in the coordinates of frame frames[j]. Returns (N,4) [frame, cx, cy, score].
    False: score >= score_thr, centre outside all GT boxes (expanded by match_margin px) and outside ignore regions."""
    out = []
    for j, i in enumerate(frames):
        d = np.asarray(dets[j], np.float32)
        d = d.reshape(len(d), -1) if d.size else np.zeros((0, 5), np.float32)   # (n,5) for 1 class or (n,6) with class
        d = d[d[:, 4] >= score_thr]
        if not len(d):
            continue
        cx, cy = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
        ok = np.ones(len(d), bool)
        for arr, m in ((anns[i].boxes, match_margin), (anns[i].ignore, 0.0)):
            for b in np.asarray(arr).reshape(-1, 4):
                ok &= ~((cx >= b[0] - m) & (cx <= b[2] + m) & (cy >= b[1] - m) & (cy <= b[3] + m))
        idx = np.nonzero(ok)[0]
        idx = idx[np.argsort(-d[idx, 4])][:per_frame]
        out += [(float(i), float(cx[k]), float(cy[k]), float(d[k, 4])) for k in idx]
    return np.array(out, np.float32).reshape(-1, 4)


@torch.no_grad()
def mine_hard_negatives(model, cache, anns, frames, mode: str, scale: float = 1.0, device="cpu", batch: int = 8,
                        score_thr: float = 0.2, per_frame: int = 10) -> np.ndarray:
    was_training = model.training
    model.eval()
    frames = list(frames)
    dets = []
    for s in range(0, len(frames), batch):
        idx = frames[s:s + batch]
        rgb, t = (x.to(device) for x in load_frames(cache, idx, t_norm=getattr(model.cfg, "tnorm", False), pre_lnorm=getattr(model.cfg, "pre_lnorm", 0)))
        if scale != 1.0:
            rgb, t = (F.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False) for x in (rgb, t))
        kw = {"both": dict(rgb=rgb, t=t), "rgb": dict(rgb=rgb), "lwir": dict(t=t)}[mode]
        for d in decode(model(**kw), model.cfg.stride, 100, score_thr):
            d = d.copy()
            d[:, :4] /= scale
            dets.append(d)
    model.train(was_training)
    return mine_from_detections(dets, anns, frames, score_thr, per_frame=per_frame)
