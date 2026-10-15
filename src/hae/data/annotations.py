"""Reading RGBTDronePerson-style COCO annotations and class mapping."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POSITIVE_CATS = (0, 1)  # "person" mode: person, rider -> a single "person" class
IGNORE_CATS = (2, 3)    # "person" mode: crowd, uncertain -> ignore regions
COCO3_CATS = (0, 1, 2)  # "coco3" mode (dataset protocol): person, rider, crowd; uncertain -> ignore


@dataclass
class FrameAnn:
    image_id: int
    file_name: str
    height: int
    width: int
    boxes: np.ndarray   # (N,4) float32 xyxy - positives
    ignore: np.ndarray  # (M,4) float32 xyxy - regions without penalty
    labels: np.ndarray = None  # (N,) int64 - class of each positive (0 person, 1 rider, 2 crowd; all 0 in person mode)


def load_annotations(path: str | Path, mode: str = "person") -> list[FrameAnn]:
    """mode="person": person+rider -> class 0, crowd/uncertain -> ignore.
    mode="coco3": 3 classes following the dataset protocol, only uncertain -> ignore."""
    if mode not in ("person", "coco3"):
        raise ValueError(f"mode must be person or coco3, got {mode!r}")
    pos_cats = POSITIVE_CATS if mode == "person" else COCO3_CATS
    d = json.loads(Path(path).read_text())
    per = {im["id"]: ([], [], []) for im in d["images"]}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:  # drop degenerate boxes
            continue
        box = [x, y, x + w, y + h]
        pos, ign, lab = per[a["image_id"]]
        is_pos = a["category_id"] in pos_cats and not a.get("ignore", 0) and not a.get("iscrowd", 0)
        if is_pos:
            pos.append(box)
            lab.append(a["category_id"] if mode == "coco3" else 0)
        else:
            ign.append(box)
    out = []
    for im in sorted(d["images"], key=lambda i: i["id"]):
        pos, ign, lab = per[im["id"]]
        out.append(FrameAnn(
            im["id"], im["file_name"], im["height"], im["width"],
            np.asarray(pos, np.float32).reshape(-1, 4),
            np.asarray(ign, np.float32).reshape(-1, 4),
            np.asarray(lab, np.int64).reshape(-1),
        ))
    return out
