"""Export a trained hae model for deployment: fp32 ONNX + int8 ONNX (static QDQ, calibrated on real training tiles at the
deployment scale; the cls/box output head stays fp32) with embedded metadata (mode, preprocessing, recommended scale and
confidence thresholds). A training checkpoint (.pt) is also written as <name>.safetensors + config.json.

    python scripts/export.py --checkpoint runs/hae-s-ir/last.pt --config configs/hae-s-ir.yaml --data data --out release/hae-s-ir
    python scripts/export.py --checkpoint models/hae-s-ir/hae-s-ir.safetensors --data data --calib-sets hituav_trainval,airesq_train \\
        --out release/hae-s-ir --conf 0.5 --sensitive-conf 0.45

Calibration sets default to the recipe's training sets (--config). Without any calibration set only fp32 is exported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "hae").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from hae.data.cache import Cache  # noqa: E402
from hae.evaluation import release_model  # noqa: E402


def export(checkpoint, out, data=None, calib_sets=(), name=None, mode=None, calib_n: int = 256, scale=None, conf=None, sensitive_conf=None,
           tile_hw=(256, 320), meta=None) -> dict:
    caches = [Cache(Path(data) / "cache", s) for s in calib_sets] if calib_sets else []
    return release_model(checkpoint, out, name=name, mode=mode, calib_caches=caches, calib_n=calib_n, scale=scale, conf=conf,
                         sensitive_conf=sensitive_conf, tile_hw=tile_hw, meta=meta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export a hae checkpoint to fp32 + int8 ONNX.")
    ap.add_argument("--checkpoint", required=True, help="training checkpoint .pt or released .safetensors (with config.json next to it)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--data", help="data root prepared by scripts/prepare_data.py (for int8 calibration)")
    ap.add_argument("--config", help="recipe YAML: its train_sets are the default calibration sets and its name the default model name")
    ap.add_argument("--calib-sets", help="comma-separated cache sets for int8 calibration (training sets of the recipe)")
    ap.add_argument("--calib-n", type=int, default=256, help="number of calibration tiles")
    ap.add_argument("--name", help="model name (default: recipe name or checkpoint stem)")
    ap.add_argument("--mode", choices=("rgb", "lwir"), help="input mode (default: from the training config)")
    ap.add_argument("--scale", type=float, help="deployment input scale (default: the recipe's eval_scale)")
    ap.add_argument("--conf", type=float, help="recommended (balanced) confidence threshold to embed")
    ap.add_argument("--sensitive-conf", type=float, help="sensitive (high-recall) confidence threshold to embed")
    ap.add_argument("--tile", default="256x320", help="export/calibration tile HxW")
    a = ap.parse_args(argv)
    recipe = None
    if a.config:
        import yaml
        recipe = yaml.safe_load(Path(a.config).read_text())
    sets = a.calib_sets.split(",") if a.calib_sets else [s["name"] for s in recipe["train_sets"]] if recipe else []
    if sets and not a.data:
        ap.error("--data is required for int8 calibration")
    th, tw = (int(v) for v in a.tile.lower().split("x"))
    rep = export(a.checkpoint, a.out, a.data, sets, a.name or (recipe or {}).get("name"), a.mode, a.calib_n, a.scale, a.conf, a.sensitive_conf, (th, tw))
    print(json.dumps(rep, indent=1, default=float))
    if not sets:
        print("no calibration sets: int8 model not exported", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
