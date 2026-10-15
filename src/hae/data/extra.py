"""Converters of open thermal aerial person datasets into the unified hae format.

Thermal output: out/thermal/<name>.jpg + out/ann.json (COCO; categories 0 person, 3 ignore), so
load_annotations(out/'ann.json') works unchanged. Cars and other vehicles are deliberately NOT labelled:
they stay background and teach the model to tell people from warm objects of a similar size.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

CATS = [{"id": 0, "name": "person"}, {"id": 3, "name": "ignore"}]


def _write(out: Path, images: list, anns: list) -> int:
    (out / "ann.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": CATS}))
    return len(images)


def convert_hituav(src_root: str | Path, out_dir: str | Path) -> int:
    """HIT-UAV (CC0). Person -> 0; Bicycle, DontCare -> ignore; Car, OtherVehicle -> background."""
    src, out = Path(src_root), Path(out_dir)
    (out / "thermal").mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for sp in ("train", "val", "test"):
        d = json.loads((src / "normal_json" / "annotations" / f"{sp}.json").read_text())
        cats = {c["id"]: c["name"].lower() for c in d["categories"]}
        new_id = {}
        for im in d["images"]:
            name = f"hituav_{sp}_{im['filename']}"
            shutil.copyfile(src / "normal_json" / sp / im["filename"], out / "thermal" / name)
            new_id[im["id"]] = len(images)
            images.append({"id": len(images), "file_name": name, "height": im["height"], "width": im["width"]})
        for a in d["annotation"]:
            kind = cats[a["category_id"]]
            cat = 0 if kind == "person" else 3 if kind in ("bicycle", "dontcare") else None
            if cat is None or a["bbox"][2] <= 0 or a["bbox"][3] <= 0:
                continue
            anns.append({"id": len(anns), "image_id": new_id[a["image_id"]], "category_id": cat, "bbox": list(a["bbox"]),
                         "ignore": 0, "iscrowd": 0})
    return _write(out, images, anns)


def convert_airesq(src_root: str | Path, out_dir: str | Path, target_wh=(640, 512)) -> int:
    """AIResQ Benchmark (CC BY 4.0): YOLO labels (class 0 = person, normalised cx cy w h).
    Frames of different resolution (640×512 and 1280×1024) are resized to target_wh; normalised boxes are unaffected."""
    src, out = Path(src_root), Path(out_dir)
    (out / "thermal").mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for p in sorted(list((src / "images").glob("*.JPG")) + list((src / "images").glob("*.jpg"))):
        img = Image.open(p)
        name = f"airesq_{p.stem}.jpg"
        if img.size != tuple(target_wh):
            img.convert("L").resize(tuple(target_wh), Image.LANCZOS).save(out / "thermal" / name, quality=95)
        else:
            shutil.copyfile(p, out / "thermal" / name)
        w, h = target_wh
        iid = len(images)
        images.append({"id": iid, "file_name": name, "height": h, "width": w})
        lab = src / "labels" / f"{p.stem}.txt"
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            parts = line.split()
            if len(parts) != 5 or int(float(parts[0])) != 0:
                continue
            cx, cy, bw, bh = (float(v) for v in parts[1:])
            if bw <= 0 or bh <= 0:
                continue
            anns.append({"id": len(anns), "image_id": iid, "category_id": 0, "ignore": 0, "iscrowd": 0,
                         "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h]})
    return _write(out, images, anns)
