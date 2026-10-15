"""Converter of the thermal part of WiSARD (MIT) into the unified hae format.

Input: folders `<date>_<Site>_<platform>_IR_<n>` (`..._VIS_...` = visible camera, ignored), frames *.jpg/*.jpeg,
YOLO labels *.txt next to them (`cls cx cy w h`, normalised; class 0 = person). Output: out/<set>/{ann.json, thermal/*.jpg}
(categories person/ignore), so `build_cache_thermal` and `load_annotations` work unchanged.

Split by SITE: no shared scene between train and val. Neighbouring frames are near duplicates, so every k-th frame with
people is taken + a share of empty frames as negatives. Sequences without any label file (only count.txt) are skipped:
their frames are NOT labelled rather than empty, so they cannot serve as negatives.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from hae.data.extra import CATS

TRAIN_SITES = ("Baker", "Carnation", "FHL")
VAL_SITES = ("Karen", "MtErie", "Airfield")
_EXT = (".jpg", ".jpeg", ".png")


def site_of(seq_name: str) -> str:
    return seq_name.split("_")[1]


def frame_index(file_name: str) -> int:
    m = re.search(r"(\d+)\.\w+$", file_name)
    if not m:
        raise ValueError(f"no frame number: {file_name}")
    return int(m.group(1))


def is_ir_sequence(seq_name: str) -> bool:
    return "_IR_" in seq_name or seq_name.endswith("_IR")


def _boxes(label: Path, w: int, h: int) -> list:
    """Person boxes (class 0) in pixels xywh; a missing file -> empty list."""
    out = []
    for line in (label.read_text().splitlines() if label.is_file() else []):
        p = line.split()
        if len(p) != 5 or int(float(p[0])) != 0:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:])
        if bw > 0 and bh > 0:
            out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h])
    return out


def _is_labelled(frames: list) -> bool:
    return any(f.with_suffix(".txt").is_file() for f in frames)


def _write(set_dir: Path, images: list, anns: list) -> None:
    (set_dir / "ann.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": CATS}))


def convert_wisard_ir(src_root: str | Path, out_root: str | Path, train_sites=TRAIN_SITES, val_sites=VAL_SITES,
                      train_every: int = 4, val_every: int = 3, neg_frac: float = 0.1) -> dict:
    """Builds out_root/wisard_ir_train and out_root/wisard_ir_val. Returns {set: {frames, boxes, negatives, sequences}}.

    For each labelled sequence: frames with people are ordered by number and every k-th is taken; negatives are evenly
    spaced frames without people (including frames without .txt), count = round(neg_frac * taken frames with people)."""
    src, out = Path(src_root), Path(out_root)
    res = {}
    for name, sites, step in (("wisard_ir_train", train_sites, train_every), ("wisard_ir_val", val_sites, val_every)):
        set_dir = out / name
        (set_dir / "thermal").mkdir(parents=True, exist_ok=True)
        images, anns, stats = [], [], {"frames": 0, "boxes": 0, "negatives": 0, "sequences": 0}
        for d in sorted(p for p in src.iterdir() if p.is_dir() and is_ir_sequence(p.name) and site_of(p.name) in sites):
            frames = sorted((f for f in d.iterdir() if f.suffix.lower() in _EXT), key=lambda f: frame_index(f.name))
            if not frames or not _is_labelled(frames):
                continue
            w, h = Image.open(frames[0]).size
            boxes = {f: _boxes(f.with_suffix(".txt"), w, h) for f in frames}
            pos = [f for f in frames if boxes[f]][::step]
            empty = [f for f in frames if not boxes[f]]
            n_neg = min(len(empty), round(neg_frac * len(pos)))
            neg = [empty[i] for i in np.linspace(0, len(empty) - 1, n_neg).round().astype(int)] if n_neg else []
            stats["sequences"] += 1
            for f in sorted(pos + neg, key=lambda f: frame_index(f.name)):
                fn = f"{d.name}_{frame_index(f.name):08d}.jpg"
                Image.open(f).convert("L").save(set_dir / "thermal" / fn, quality=95)
                iid = len(images)
                images.append({"id": iid, "file_name": fn, "height": h, "width": w})
                for b in boxes[f]:
                    anns.append({"id": len(anns), "image_id": iid, "category_id": 0, "bbox": b, "ignore": 0, "iscrowd": 0})
                stats["boxes"] += len(boxes[f])
            stats["negatives"] += len(neg)
        stats["frames"] = len(images)
        _write(set_dir, images, anns)
        res[name] = stats
    return res
