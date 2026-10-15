"""COCO-style AP via pycocotools. Ignore regions are passed as iscrowd."""
from __future__ import annotations

import contextlib
import io

import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _xywh(b):
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def evaluate_detections(gt_boxes, gt_ignore, dets, img_hw) -> dict:
    h, w = img_hw
    images = [{"id": i, "height": h, "width": w} for i in range(len(gt_boxes))]
    anns, aid = [], 1
    for i, (g, ig) in enumerate(zip(gt_boxes, gt_ignore)):
        for crowd, arr in ((0, g), (1, ig)):
            for b in arr:
                bb = _xywh(b)
                anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": bb,
                             "area": bb[2] * bb[3], "iscrowd": crowd})
                aid += 1
    res = [{"image_id": i, "category_id": 1, "bbox": _xywh(d), "score": float(d[4])}
           for i, ds in enumerate(dets) for d in ds]
    zero = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0}
    if not res or not any(a["iscrowd"] == 0 for a in anns):
        return zero
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "person"}]}
        gt.createIndex()
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
    # Computed directly from ev.eval: summarize() looks for maxDets==100 and returns -1 with [1,10,300]
    prec = ev.eval["precision"][:, :, 0, 0, -1]   # [T(IoU), R(recall)] for area=all, maxDets=300
    rec = ev.eval["recall"][:, 0, 0, -1]          # [T]

    def _mean(a):
        a = a[a > -1]
        return float(a.mean()) if a.size else 0.0

    return {"AP": _mean(prec), "AP50": _mean(prec[0]), "AP75": _mean(prec[5]), "AR": _mean(rec)}


CLASS_NAMES = ("person", "rider", "crowd")
_IOU_THRS = np.linspace(0.25, 0.95, 15)      # 0.25 ... 0.95 step 0.05: [0] = 0.25, [5] = 0.50, [10] = 0.75
TINY_AREA = 20.0 ** 2                        # "tiny" in the dataset protocol: size < 20 px


def _mean_valid(a) -> float:
    a = np.asarray(a, float)
    a = a[a > -1]
    return float(a.mean()) if a.size else float("nan")


def evaluate_protocol(gt_boxes, gt_labels, gt_ignore, dets, img_hw, class_names=CLASS_NAMES) -> dict:
    """RGBTDronePerson protocol: AP per class, averaged over classes present in GT;
    mAP50, mAP25, mAP75, mAP@[.5:.95], mAP50^tiny (sqrt(area) < 20 px). "uncertain" = ignore for ALL classes.
    dets: list of (n,6) [x1,y1,x2,y2,score,cls]."""
    h, w = img_hw
    k = len(class_names)
    images = [{"id": i, "height": h, "width": w} for i in range(len(gt_boxes))]
    anns, aid = [], 1
    for i, (g, lab, ig) in enumerate(zip(gt_boxes, gt_labels, gt_ignore)):
        for b, c in zip(g, lab):
            bb = _xywh(b)
            anns.append({"id": aid, "image_id": i, "category_id": int(c) + 1, "bbox": bb, "area": bb[2] * bb[3], "iscrowd": 0})
            aid += 1
        for b in ig:                                  # in COCOeval iscrowd only applies within its own class -> one copy per class
            bb = _xywh(b)
            for c in range(k):
                anns.append({"id": aid, "image_id": i, "category_id": c + 1, "bbox": bb, "area": bb[2] * bb[3], "iscrowd": 1})
                aid += 1
    res = [{"image_id": i, "category_id": int(d[5]) + 1, "bbox": _xywh(d), "score": float(d[4])}
           for i, ds in enumerate(dets) for d in ds]
    zero = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AP25": 0.0, "AR": 0.0, "AP50_tiny": 0.0, "per_class": {}}
    if not res or not any(a["iscrowd"] == 0 for a in anns):
        return zero
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns,
                      "categories": [{"id": c + 1, "name": n} for c, n in enumerate(class_names)]}
        gt.createIndex()
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.params.iouThrs = _IOU_THRS
        ev.params.areaRng = [[0, 1e10], [0, TINY_AREA]]
        ev.params.areaRngLbl = ["all", "tiny"]
        ev.evaluate()
        ev.accumulate()
    prec, rec = ev.eval["precision"], ev.eval["recall"]           # [T,R,K,A,M], [T,K,A,M]
    present = [c for c in range(k) if any((np.asarray(l) == c).any() for l in gt_labels)]
    def per_class(t_idx, area=0):
        return {class_names[c]: _mean_valid(np.mean([_mean_valid(prec[t, :, c, area, -1]) for t in t_idx])) for c in present}
    def mean_cls(d):
        return _mean_valid(list(d.values())) if d else 0.0
    ap50 = per_class([5])
    return {"AP": mean_cls({n: np.nanmean([_mean_valid(prec[t, :, c, 0, -1]) for t in range(5, 15)]) for n, c in zip((class_names[c] for c in present), present)}),
            "AP50": mean_cls(ap50), "AP75": mean_cls(per_class([10])), "AP25": mean_cls(per_class([0])),
            "AR": float(np.nanmean([_mean_valid(rec[t, c, 0, -1]) for t in range(5, 15) for c in present])),
            "AP50_tiny": mean_cls(per_class([5], area=1)), "per_class": ap50}
