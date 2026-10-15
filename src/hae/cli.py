"""The `hae` command line: models | info | predict | download | export. models/info/predict need only numpy, opencv, onnx, onnxruntime."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _need(path: str, what: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{what} not found: {p}")
    return p


def cmd_models(a) -> int:
    from hae import ui
    from hae.detector import list_models, models_root, resolve_model
    from hae.infer import read_metadata
    rows = []
    for n in list_models(a.root):
        p = resolve_model(n, a.root)
        m = read_metadata(p)
        rows.append([n, m.get("mode", "?"), f"{p.stat().st_size // 1024} KB", f"x{m.get('recommended_scale', 1.0)}", m.get("recommended_conf", "-"), m.get("sensitive_conf", "-"), ", ".join(m.get("class_names", []))])
    ui.table(f"models ({models_root(a.root)})", ["model", "mode", "int8 size", "scale", "conf (balanced)", "conf (sensitive)", "classes"], rows or [["(none)", "", "", "", "", "", ""]])
    return 0


def cmd_info(a) -> int:
    from hae import ui
    from hae.detector import resolve_model
    from hae.infer import read_metadata
    p = resolve_model(a.model, a.root)
    meta = read_metadata(p)
    keys = ("name", "hae_version", "mode", "tnorm", "stride", "ncls", "class_names", "recommended_scale", "recommended_conf", "sensitive_conf", "preprocess", "recipe", "license")
    ui.banner(f"hae · {p.name}", {k: meta[k] for k in keys if k in meta})
    return 0


def _next_dir(base: Path) -> Path:
    """runs/predict/exp, exp2, exp3, ...: new results never overwrite old ones."""
    if not base.exists():
        return base
    k = 2
    while base.with_name(f"{base.name}{k}").exists():
        k += 1
    return base.with_name(f"{base.name}{k}")


def cmd_predict(a) -> int:
    from hae.detector import Detector
    src = Path(a.source)
    if not src.exists():
        raise SystemExit(f"source not found: {src}")
    det = Detector(a.model, conf=a.conf, preset=a.preset, scale=a.scale, tile=tuple(int(v) for v in a.tile.split(",")) if a.tile else None, overlap=a.overlap, root=a.root)
    res = det.predict(src)
    res = res if isinstance(res, list) else [res]
    out = _next_dir(Path(a.project) / "predict" / "exp")
    if not a.no_save:
        out.mkdir(parents=True, exist_ok=True)
    from hae import ui
    ui.banner(f"hae · predict · {det.path.stem}", {"Mode": det.mode, "Confidence threshold": det.conf, "Input scale": f"x{det.scale}", "Tiles (SAHI)": a.tile or "no", "Images": len(res)})
    allj = {}
    for k, r in enumerate(res):
        print(" ", r.summary())
        name = Path(r.path).name if r.path else f"{k}.png"
        if not a.no_save:
            r.save(out / name, thr=det.conf)
        allj[name] = json.loads(r.to_json())
    if a.json and not a.no_save:
        (out / "results.json").write_text(json.dumps(allj, ensure_ascii=False, indent=1))
    if res and res[0].speed:
        mean = {k: sum(r.speed[k] for r in res) / len(res) for k in res[0].speed}
        print(f"speed per image: preprocess {mean['preprocess']:.1f} · inference {mean['inference']:.1f} · postprocess {mean['postprocess']:.1f} ms")
    print(f"total detections: {sum(len(r) for r in res)}")
    if not a.no_save:
        print("results:", out)
    return 0


def cmd_export(a) -> int:
    from hae.export import export_onnx
    from hae.checkpoint import load_checkpoint
    model, _ = load_checkpoint(_need(a.checkpoint, "checkpoint"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{a.name}_{a.mode}.onnx"
    print(export_onnx(model, p, a.mode, tuple(int(v) for v in a.hw.split(",")))["path"])
    return 0


def cmd_download(a) -> int:
    from hae import hub
    names = list(hub.RELEASED_MODELS) if a.all else a.names
    if not names:
        raise SystemExit("give model names (e.g. hae-s-ir) or --all")
    try:
        for n in names:
            print(n, "->", hub.download(n, int8=not a.fp32, revision=a.revision))
    except ImportError as e:
        raise SystemExit(str(e))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hae", description="hae: tiny person detectors for search-and-rescue drones (thermal + RGB)")
    ap.add_argument("--root", default=None, help="model directory (default: $HAE_MODELS, ./models, the repository models/, then the Hugging Face Hub)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list locally available models").set_defaults(f=cmd_models)
    i = sub.add_parser("info", help="model metadata (name or .onnx path)"); i.add_argument("model"); i.set_defaults(f=cmd_info)
    p = sub.add_parser("predict", help="detect people in an image, a folder or a list of images"); p.set_defaults(f=cmd_predict)
    p.add_argument("model", help='model name ("hae-s-ir") or path to an .onnx file'); p.add_argument("source", help="image or folder")
    p.add_argument("--conf", type=float, default=None, help="confidence threshold (default: the one recommended in the model metadata)")
    p.add_argument("--preset", choices=["balanced", "sensitive"], default="balanced", help="balanced = max F1; sensitive = max F2 (higher recall, more false alarms)")
    p.add_argument("--scale", type=float, default=None, help="input scale (default: from the model metadata)")
    p.add_argument("--tile", default=None, help='SAHI tile "H,W", e.g. 256,320'); p.add_argument("--overlap", type=float, default=0.2, help="tile overlap fraction")
    p.add_argument("--project", default="runs", help="output location: <project>/predict/exp[N]")
    p.add_argument("--json", action="store_true", help="also write results.json"); p.add_argument("--no-save", action="store_true", help="do not write images")
    d = sub.add_parser("download", help='download released models from the Hugging Face Hub (needs pip install "hae[hub]")'); d.set_defaults(f=cmd_download)
    d.add_argument("names", nargs="*", help="model names, e.g. hae-s-ir"); d.add_argument("--all", action="store_true", help="all released models")
    d.add_argument("--fp32", action="store_true", help="fp32 model instead of int8"); d.add_argument("--revision", default=None, help="Hub revision (branch, tag or commit)")
    e = sub.add_parser("export", help="checkpoint (.pt, or .safetensors with config.json) -> ONNX (needs PyTorch)"); e.set_defaults(f=cmd_export)
    e.add_argument("checkpoint", help="a .pt checkpoint, or a .safetensors file with a sibling config.json"); e.add_argument("out", help="output directory"); e.add_argument("--name", default="hae", help="file name prefix")
    e.add_argument("--mode", default="lwir", choices=["rgb", "lwir", "both"]); e.add_argument("--hw", default="256,320", help='dummy input "H,W" for tracing')
    a = ap.parse_args(argv)
    try:
        return a.f(a)
    except FileNotFoundError as err:
        raise SystemExit(f"error: {err}")


if __name__ == "__main__":
    sys.exit(main())
