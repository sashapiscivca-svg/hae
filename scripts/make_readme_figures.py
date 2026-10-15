"""Draw the README figures (docs/assets/hae-hero.jpg, docs/assets/hae-examples.jpg) from HIT-UAV test frames.

Needs the prepared `hituav_test` set (scripts/prepare_data.py hituav ...) and opencv. Green boxes: hae-s-ir int8 detections
at the balanced threshold, with scores; orange circles: people in the ground truth that the model missed (IoU < 0.5).
The frames are fixed by id so the figure is reproducible; one hard daytime frame is included on purpose.

    python scripts/make_readme_figures.py --data data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "hae").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from hae.detector import Detector  # noqa: E402

HERO, EXAMPLES = 220, (59, 469, 312, 9)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def render(det: Detector, frame: np.ndarray, gt: list, scale: int):
    d = det.detect(frame)
    used: set[int] = set()
    for box in d:
        best, best_iou = -1, 0.5
        for j, g in enumerate(gt):
            if j not in used and (v := iou(box[:4], g)) >= best_iou:
                best, best_iou = j, v
        if best >= 0:
            used.add(best)
    missed = [g for j, g in enumerate(gt) if j not in used]
    big = cv2.resize(frame, (frame.shape[1] * scale, frame.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    img = cv2.applyColorMap(big, cv2.COLORMAP_INFERNO)   # white-hot grayscale shown with a heat palette for readability
    for x1, y1, x2, y2, s, _ in d:
        p1, p2 = (int(x1 * scale) - 3, int(y1 * scale) - 3), (int(x2 * scale) + 3, int(y2 * scale) + 3)
        cv2.rectangle(img, p1, p2, (80, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(img, f"{s:.2f}", (p1[0], p1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale / 2 + 0.15, (80, 255, 80), 1, cv2.LINE_AA)
    for x1, y1, x2, y2 in missed:
        c = (int((x1 + x2) / 2 * scale), int((y1 + y2) / 2 * scale))
        cv2.circle(img, c, int(max(x2 - x1, y2 - y1) * scale * 0.9) + 6, (0, 200, 255), 2, cv2.LINE_AA)
    return img, len(gt) - len(missed), len(gt)


def caption(img: np.ndarray, text: str, height: int, size: float) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], height), (0, 0, 0), -1)
    cv2.putText(img, text, (10, int(height * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1, cv2.LINE_AA)


def describe(file_name: str) -> str:
    # HIT-UAV names: <split>_<time 0 day / 1 night>_<altitude m>_<camera angle deg>_...
    p = file_name.split("_")
    return f"{'night' if p[2] == '1' else 'day'}, {p[3]} m, camera {p[4]} deg"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data", help="data root prepared by scripts/prepare_data.py")
    ap.add_argument("--model", default=str(ROOT / "models/hae-s-ir/hae-s-ir_lwir_int8.onnx"))
    ap.add_argument("--out", default=str(ROOT / "docs/assets"))
    a = ap.parse_args()
    cache = Path(a.data) / "cache"
    meta = json.loads((cache / "hituav_test_meta.json").read_text())
    frames = np.memmap(cache / "hituav_test_t.u8", np.uint8, "r", shape=(meta["n"], meta["h"], meta["w"]))
    ann = json.loads((cache / "hituav_test_ann.json").read_text())
    names = {im["id"]: im["file_name"] for im in ann["images"]}
    gts: dict[int, list] = {i: [] for i in names}
    for an in ann["annotations"]:
        if an["category_id"] == 0:
            x, y, w, h = an["bbox"]
            gts[an["image_id"]].append([x, y, x + w, y + h])
    det = Detector(a.model, threads=4)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    hero, found, total = render(det, np.asarray(frames[HERO]), gts[HERO], 2)
    caption(hero, f"HIT-UAV test (CC0) | {describe(names[HERO])} | hae-s-ir int8 | found {found} of {total} people", 34, 0.7)
    cv2.imwrite(str(out / "hae-hero.jpg"), hero, [cv2.IMWRITE_JPEG_QUALITY, 88])

    tiles = []
    for i in EXAMPLES:
        img, found, total = render(det, np.asarray(frames[i]), gts[i], 1)
        caption(img, f"{describe(names[i])}: found {found}/{total}", 26, 0.55)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    cv2.imwrite(str(out / "hae-examples.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out / "hae-hero.jpg", out / "hae-examples.jpg")


if __name__ == "__main__":
    main()
