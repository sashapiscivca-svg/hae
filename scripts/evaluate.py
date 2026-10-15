"""Evaluate an exported hae ONNX model on sets prepared by scripts/prepare_data.py.

Reports AP50 (COCO-style, IoU 0.5) and, at the model's confidence presets, recall / precision / F1 / false positives per frame
(class-agnostic greedy matching at IoU >= 0.5; detections inside ignore regions are not counted as false).
Preprocessing, input scale and presets are read from the model's ONNX metadata unless overridden.

    python scripts/evaluate.py --model models/hae-s-ir/hae-s-ir_lwir_int8.onnx --data data --sets hituav_test,airesq_val,wisard_ir_val
    python scripts/evaluate.py --model models/hae-s-ir/hae-s-ir_lwir.onnx --data data --sets hituav_test --conf 0.3,0.4 --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "hae").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from hae.data.annotations import load_annotations  # noqa: E402
from hae.data.cache import Cache, ensure_lnorm_cache  # noqa: E402
from hae.evaluation import detect_onnx, prf_table  # noqa: E402
from hae.infer import read_metadata  # noqa: E402
from hae.metrics import evaluate_detections  # noqa: E402


def evaluate_model(model, data, sets, scale=None, confs=None, stride_frames: int = 1) -> dict:
    meta = read_metadata(model)
    mode = meta.get("mode", "lwir")
    if mode not in ("rgb", "lwir"):
        raise ValueError(f"single-mode models only (rgb or lwir), got {mode!r}")
    scale = float(meta.get("recommended_scale", 1.0) if scale is None else scale)
    presets = {"balanced": meta.get("recommended_conf"), "sensitive": meta.get("sensitive_conf")}
    thrs = list(confs) if confs else [v for v in presets.values() if v is not None] or [0.25]
    cache_dir = Path(data) / "cache"
    pre = int(meta.get("pre_lnorm", 0) or 0)
    if pre:
        cache_dir = ensure_lnorm_cache(cache_dir, pre)
    out = {"model": str(model), "mode": mode, "scale": scale, "presets": presets, "sets": {}}
    for name in sets:
        c = Cache(cache_dir, name)
        anns = load_annotations(Path(data) / "cache" / f"{name}_ann.json", "person")
        frames = list(range(0, c.n, max(1, stride_frames)))
        dets = detect_onnx(model, c, frames, mode, bool(meta.get("tnorm", False)), int(meta.get("stride", 4)), scale, 0.01, pre)
        gts, igs = [anns[i].boxes for i in frames], [anns[i].ignore for i in frames]
        out["sets"][name] = {"frames": len(frames), "gt": int(sum(len(g) for g in gts)),
                             "AP50": float(evaluate_detections(gts, igs, dets, (c.h, c.w))["AP50"]), "at": prf_table(gts, igs, dets, thrs)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a hae ONNX model on prepared sets.")
    ap.add_argument("--model", required=True, help="ONNX file (fp32 or int8)")
    ap.add_argument("--data", required=True, help="data root prepared by scripts/prepare_data.py")
    ap.add_argument("--sets", required=True, help="comma-separated set names, e.g. hituav_test,airesq_val")
    ap.add_argument("--scale", type=float, help="input scale (default: recommended_scale from the metadata)")
    ap.add_argument("--conf", help="comma-separated confidence thresholds (default: the model's presets)")
    ap.add_argument("--every", type=int, default=1, help="evaluate every k-th frame (quick checks)")
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)
    confs = [float(v) for v in a.conf.split(",")] if a.conf else None
    rep = evaluate_model(a.model, a.data, a.sets.split(","), a.scale, confs, a.every)
    print(f"{Path(a.model).name}: mode {rep['mode']}, scale x{rep['scale']:g}, presets {rep['presets']}")
    print(f"{'set':20} {'frames':>6} {'gt':>6} {'AP50':>6} {'conf':>5} {'recall':>7} {'prec':>6} {'F1':>6} {'FP/frame':>8}")
    for name, r in rep["sets"].items():
        for t, m in r["at"].items():
            print(f"{name:20} {r['frames']:6} {r['gt']:6} {r['AP50']:6.3f} {float(t):5.2f} {m['recall']:7.3f} {m['precision']:6.3f} {m['f1']:6.3f} {m['fp_per_frame']:8.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
