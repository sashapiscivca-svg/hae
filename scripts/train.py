"""Train a hae model from a YAML recipe on data prepared by scripts/prepare_data.py.

    python scripts/train.py --config configs/hae-s-ir.yaml --data data --out runs/hae-s-ir
    python scripts/train.py --config configs/hae-s-ir.yaml --data data --out runs/hae-s-ir --device cuda --resume

Recipe format (see configs/*.yaml):
    name: hae-s-ir
    train_sets: [{name: hituav_trainval, weight: 0.6}, {name: airesq_train, weight: 0.4}]   # first = main set
    val_sets: [hituav_test, airesq_val, wisard_ir_val]   # first = in-training validation; all = final evaluation
    eval_stride: 4                                        # in-training validation uses every k-th frame of the first val set
    train: {size: s, epochs: 90, ...}                     # hae.train.TrainConfig fields

Outputs in --out: last.pt / best.pt / resume.pt, log.jsonl, results.csv/png, dashboard.png, final.json (metrics of the final EMA
model on every val set: thermal sets in mode lwir, RGB sets in mode rgb, at the recipe's eval_scale).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "hae").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from hae.data.annotations import load_annotations  # noqa: E402
from hae.data.cache import Cache  # noqa: E402
from hae.train import TrainConfig, ann_path, evaluate, train  # noqa: E402


def load_recipe(path: str | Path) -> dict:
    import yaml
    r = yaml.safe_load(Path(path).read_text())
    for k in ("train_sets", "val_sets", "train"):
        if k not in r:
            raise ValueError(f"{path}: missing key {k!r}")
    w = sum(float(s["weight"]) for s in r["train_sets"])
    if abs(w - 1.0) > 1e-6:
        raise ValueError(f"{path}: train_sets weights must sum to 1, got {w:g}")
    return r


def build_config(recipe: dict, data: str | Path, out: str | Path, **over) -> TrainConfig:
    """TrainConfig of a recipe: the first train set is the main set (weight = 1 - sum of the others), the others are extras."""
    cache = Path(data) / "cache"
    sets = list(recipe["train_sets"]) + [{"name": v} for v in recipe["val_sets"]]
    missing = [s["name"] for s in sets if not (cache / f"{s['name']}_meta.json").exists()]
    if missing:
        raise FileNotFoundError(f"sets not prepared in {cache}: {missing} (run scripts/prepare_data.py first)")
    val = recipe["val_sets"][0]
    n = json.loads((cache / f"{val}_meta.json").read_text())["n"]
    kw = {**recipe["train"], **{k: v for k, v in over.items() if v is not None}}
    return TrainConfig(cache_dir=str(cache), ann_root=str(Path(data)), out_dir=str(out), train_set=recipe["train_sets"][0]["name"],
                       extra=[{"name": s["name"], "weight": float(s["weight"])} for s in recipe["train_sets"][1:]] or None,
                       eval_split=val, eval_frames=list(range(0, n, int(recipe.get("eval_stride", 1)))), **kw)


def final_eval(model, cfg: TrainConfig, val_sets) -> dict:
    """Metrics of the model on every val set: lwir for thermal-only caches, rgb for RGB caches."""
    cache_dir = cfg.cache_dir
    if cfg.pre_lnorm:
        from hae.data.cache import ensure_lnorm_cache
        cache_dir = str(ensure_lnorm_cache(cache_dir, cfg.pre_lnorm))
    res = {}
    for name in val_sets:
        c = Cache(cache_dir, name)
        mode = "lwir" if c.rgb is None else "rgb"
        anns = load_annotations(ann_path(cfg, name), "coco3" if cfg.classes == 3 else "person")
        res[f"{mode}@{name}"] = evaluate(model, c, anns, mode=mode, scale=cfg.eval_scale)
    return res


def run(recipe: dict, data, out, **over) -> dict:
    cfg = build_config(recipe, data, out, **over)
    t0 = time.time()
    res = train(cfg)
    final = final_eval(res["model"], cfg, recipe["val_sets"])
    (Path(out) / "final.json").write_text(json.dumps(final, indent=1, default=float))
    if cfg.verbose:
        for k, r in final.items():
            print(f"VAL {k}: " + ", ".join(f"{m}={v:.3f}" for m, v in r.items() if not isinstance(v, dict)), flush=True)
        print(f"done in {time.time() - t0:.0f} s -> {out}", flush=True)
    return {"cfg": cfg, "final": final, **res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a hae model from a YAML recipe.")
    ap.add_argument("--config", required=True, help="recipe YAML, e.g. configs/hae-s-ir.yaml")
    ap.add_argument("--data", required=True, help="data root prepared by scripts/prepare_data.py")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--epochs", type=int, help="override the recipe's number of epochs")
    ap.add_argument("--crops-per-epoch", type=int, help="override crops per epoch (quick experiments)")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch")
    ap.add_argument("--prefetch", type=int, default=0, help="background threads preparing batches")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/resume.pt")
    ap.add_argument("--init", help="initial weights for fine-tuning: a released .safetensors (with config.json) or a .pt")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    run(load_recipe(a.config), a.data, a.out, epochs=a.epochs, crops_per_epoch=a.crops_per_epoch, device=a.device, seed=a.seed,
        threads=a.threads, prefetch=a.prefetch, resume=a.resume, verbose=not a.quiet,
        **({"init_weights": a.init} if a.init else {}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
