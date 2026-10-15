"""Operating-point recall and false alarms per frame at a given confidence threshold.

COCO AR averages recall over IoU 0.5..0.95 and does not answer the rescuer's question "how many people will I find and how many
false alarms will I get". Here: class-agnostic greedy matching by confidence at IoU >= iou_thr; detections inside ignore regions
are not counted as false.
"""
from __future__ import annotations

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    iy = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = ix * iy
    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ub = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (ua[:, None] + ub[None, :] - inter + 1e-9)


def recall_at(gts, ignores, dets, thr: float, iou_thr: float = 0.5) -> dict:
    tp = fp = n_gt = 0
    for g, ig, d in zip(gts, ignores, dets):
        g = np.asarray(g, np.float32).reshape(-1, 4)
        d = np.asarray(d, np.float32)
        d = d.reshape(len(d), -1) if d.size else np.zeros((0, 5), np.float32)
        d = d[d[:, 4] >= thr] if len(d) else d
        d = d[np.argsort(-d[:, 4])] if len(d) else d
        n_gt += len(g)
        used = np.zeros(len(g), bool)
        for det in d:
            box = det[None, :4]
            hit = -1
            if len(g):
                iou = _iou(box, g)[0]
                iou[used] = 0
                j = int(np.argmax(iou))
                hit = j if iou[j] >= iou_thr else -1
            if hit >= 0:
                used[hit] = True
                tp += 1
                continue
            cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
            in_ign = any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in np.asarray(ig).reshape(-1, 4))
            if not in_ign:
                fp += 1
    n = max(len(gts), 1)
    return {"gt": n_gt, "tp": tp, "fp": fp, "recall": tp / max(n_gt, 1), "fp_per_frame": fp / n, "precision": tp / max(tp + fp, 1)}


def _center_nms(d: np.ndarray, radius: float) -> np.ndarray:
    """Duplicates of one detection (centres closer than radius px) count as one alarm: the most confident one is kept."""
    if not len(d):
        return d
    d = d[np.argsort(-d[:, 4])]
    c = np.stack([(d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2], 1)
    keep = []
    for i in range(len(d)):
        if all(np.hypot(*(c[i] - c[j])) > radius for j in keep):
            keep.append(i)
    return d[keep]


def recall_center(gts, ignores, dets, thr: float, margin: float = 2.0, cluster_px: float = 8.0) -> dict:
    """Operator-oriented metric: a hit is a detection centre inside a labelled person (+margin px); duplicates within cluster_px are one alarm.
    Unlike IoU >= 0.5 it does not penalise an imprecise box on a real person."""
    tp = fp = n_gt = 0
    for g, ig, d in zip(gts, ignores, dets):
        g = np.asarray(g, np.float32).reshape(-1, 4)
        d = np.asarray(d, np.float32)
        d = d.reshape(len(d), -1) if d.size else np.zeros((0, 5), np.float32)
        d = _center_nms(d[d[:, 4] >= thr] if len(d) else d, cluster_px)
        n_gt += len(g)
        used = np.zeros(len(g), bool)
        for det in d:
            cx, cy = (det[0] + det[2]) / 2, (det[1] + det[3]) / 2
            hit = -1
            for j, b in enumerate(g):
                if not used[j] and b[0] - margin <= cx <= b[2] + margin and b[1] - margin <= cy <= b[3] + margin:
                    hit = j
                    break
            if hit >= 0:
                used[hit] = True
                tp += 1
            elif not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in np.asarray(ig).reshape(-1, 4)):
                fp += 1
    n = max(len(gts), 1)
    return {"gt": n_gt, "tp": tp, "fp": fp, "recall": tp / max(n_gt, 1), "fp_per_frame": fp / n, "precision": tp / max(tp + fp, 1)}


def frame_level(gts, ignores, dets, thr: float, margin: float = 2.0) -> dict:
    """Frame level: share of frames with people where at least one is found (frame_recall), and share of empty frames with a false alarm (false_alarm_frames)."""
    pos = hit = neg = alarm = 0
    for g, ig, d in zip(gts, ignores, dets):
        r = recall_center([g], [ig], [d], thr, margin)
        if len(np.asarray(g).reshape(-1, 4)):
            pos += 1
            hit += int(r["tp"] > 0)
        else:
            neg += 1
            alarm += int(r["fp"] > 0)
    return {"frames_with_people": pos, "frame_recall": hit / max(pos, 1), "empty_frames": neg, "false_alarm_frames": alarm / max(neg, 1)}


def choose_presets(sets: dict, beta_sensitive: float = 2.0) -> dict:
    """Threshold presets by F1 averaged over datasets (balanced) and F-beta, beta=2 (sensitive: recall weighs twice as much as precision)."""
    thr = sorted({float(t) for d in sets.values() for t in d})

    def fb(t, b):
        v = []
        for d in sets.values():
            m = d[str(t)]
            p, r = m["precision"], m["recall"]
            v.append((1 + b * b) * p * r / max(b * b * p + r, 1e-9))
        return sum(v) / len(v)
    return {"balanced": max(thr, key=lambda t: fb(t, 1.0)), "sensitive": max(thr, key=lambda t: fb(t, beta_sensitive))}
