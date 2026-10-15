"""Temporal confirmation of detections in video: an object is accepted only if it appears in at least k of the last n frames
(including the current one; a detection within radius px of its centre). Causal filter: uses past frames only, so it works in real time."""
from __future__ import annotations

import numpy as np


def _centers(d: np.ndarray) -> np.ndarray:
    return np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2], 1) if len(d) else np.zeros((0, 2), np.float32)


def temporal_confirm(seq: list, k: int = 2, n: int = 3, radius: float = 10.0, min_score: float = 0.0) -> list:
    """seq: list of (m,6) arrays, one per frame of a video. Returns a list of filtered arrays of the same length.
    At the start of a video confirmation is impossible for the first k-1 frames, so objects appear with a delay of k-1 frames."""
    seq = [np.asarray(d, np.float32).reshape(len(d), -1) if np.asarray(d).size else np.zeros((0, 6), np.float32) for d in seq]
    out = []
    for t, d in enumerate(seq):
        if not len(d):
            out.append(d)
            continue
        c = _centers(d)
        hits = np.ones(len(d), int)                                    # current frame
        for p in range(max(0, t - n + 1), t):
            prev = seq[p][seq[p][:, 4] >= min_score] if len(seq[p]) else seq[p]
            if not len(prev):
                continue
            pc = _centers(prev)
            dist = np.hypot(c[:, None, 0] - pc[None, :, 0], c[:, None, 1] - pc[None, :, 1])
            hits += (dist.min(1) <= radius).astype(int)
        out.append(d[hits >= k])
    return out
