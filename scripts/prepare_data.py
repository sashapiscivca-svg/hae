"""Prepare the open thermal datasets used by the released hae model (hae-s-ir): convert the original downloads into the
unified layout and build the uint8 frame caches the trainer reads.

    python scripts/prepare_data.py sources                                  # download URLs and licences
    python scripts/prepare_data.py hituav --src ~/dl/HIT-UAV --out data     # training (trainval) + evaluation (test)
    python scripts/prepare_data.py airesq --src ~/dl/AIResQ --out data      # training (train) + evaluation (val)
    python scripts/prepare_data.py wisard-ir --src ~/dl/WiSARD --out data   # evaluation only (val)

Nothing is downloaded automatically: fetch each dataset yourself from the URL printed by `sources` and accept its licence.

Layout produced under --out (<data>):
    <data>/work/<dataset>/...            converter output (all frames of the download, the converter's own split)
    <data>/sets/<set>/ann.json           COCO annotations of one split (categories 0 person, 3 ignore)
    <data>/sets/<set>/thermal/           frames of that split (hard links into work/ when possible)
    <data>/cache/<set>_t.u8, <set>_meta.json, <set>_ann.json   caches used by scripts/train.py and scripts/evaluate.py
Set names: hituav_trainval, hituav_test, airesq_train, airesq_val, wisard_ir_train, wisard_ir_val.

Alignment (default). The exact COCO annotation files of every split used to train and evaluate the released model are
published in `data/annotations/<set>.json` of this repository. By default each split is ALIGNED to them: only the images
listed there are kept (matched by file_name) and the published annotations are used; images that the conversion did not
produce are reported as missing, and frames whose converted labels differ from the published ones are counted.
`--no-align` instead keeps the converters' own split (reproduction from scratch). Splits without a published file
(wisard_ir_train) always use the converter's split.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src" / "hae").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

from hae.data.cache import build_cache_thermal  # noqa: E402
from hae.data.extra import CATS  # noqa: E402

PUBLISHED = ROOT / "data" / "annotations"

SOURCES = {
    "hituav": {"title": "HIT-UAV: high-altitude infrared thermal dataset", "license": "CC0-1.0",
               "url": "https://zenodo.org/records/7633134", "sets": ("hituav_trainval", "hituav_test"),
               "src": "unpacked archive containing normal_json/{annotations,train,val,test}"},
    "airesq": {"title": "AIResQ Benchmark (thermal)", "license": "CC-BY-4.0",
               "url": "https://zenodo.org/records/17405074", "sets": ("airesq_train", "airesq_val"),
               "src": "folder with images/*.jpg and YOLO labels/*.txt"},
    "wisard-ir": {"title": "WiSARD, thermal (IR) sequences", "license": "MIT",
                  "url": "https://sites.google.com/uw.edu/wisard", "sets": ("wisard_ir_train", "wisard_ir_val"),
                  "src": "folder with <date>_<Site>_<platform>_IR_<n>/ sequence folders (frames + YOLO .txt)"},
}


def sources_text() -> str:
    lines = ["Datasets used by the released hae model (download them yourself; nothing is fetched automatically):"]
    for k, s in SOURCES.items():
        lines.append(f"  {k:13} {s['title']}\n  {'':13} licence {s['license']} · {s['url']}\n  {'':13} --src: {s['src']}")
    return "\n".join(lines)


# ───────────────────────── conversion: dataset -> candidate splits ─────────────────────────
# Each converter returns {set name: (COCO dict, image directory)} with the converter's own split.

def _subset(d: dict, keep) -> dict:
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    imgs, anns = [], []
    for im in d["images"]:
        if not keep(im["file_name"]):
            continue
        iid = len(imgs)
        imgs.append({**im, "id": iid})
        anns += [{**a, "id": len(anns) + k, "image_id": iid} for k, a in enumerate(by.get(im["id"], []))]
    return {"images": imgs, "annotations": anns, "categories": d.get("categories", CATS)}


def _airesq_group(fn: str) -> str:
    """Recording (hour of capture or clip number) of an AIResQ frame: whole recordings are held out."""
    m = re.search(r"DJI_(\d{14})_", fn)
    return "t" + m.group(1)[:10] if m else re.search(r"DJI_(\d+)", fn).group(0)


def convert(dataset: str, src: Path, work: Path) -> dict:
    from hae.data import extra
    w = work / dataset
    if dataset == "hituav":
        extra.convert_hituav(src, w)
        d = json.loads((w / "ann.json").read_text())
        return {"hituav_trainval": (_subset(d, lambda f: f.split("_")[1] in ("train", "val")), w / "thermal"),
                "hituav_test": (_subset(d, lambda f: f.split("_")[1] == "test"), w / "thermal")}
    if dataset == "airesq":
        extra.convert_airesq(src, w)
        d = json.loads((w / "ann.json").read_text())
        groups = sorted({_airesq_group(i["file_name"]) for i in d["images"]})
        val = set(groups[3::7])   # ~14 % of the recordings are held out entirely
        return {"airesq_train": (_subset(d, lambda f: _airesq_group(f) not in val), w / "thermal"),
                "airesq_val": (_subset(d, lambda f: _airesq_group(f) in val), w / "thermal")}
    if dataset == "wisard-ir":
        from hae.data.wisard_ir import convert_wisard_ir
        convert_wisard_ir(src, w)
        return {n: (json.loads((w / n / "ann.json").read_text()), w / n / "thermal") for n in ("wisard_ir_train", "wisard_ir_val")}
    raise KeyError(f"unknown dataset {dataset!r}; known: {', '.join(SOURCES)}")


# ───────────────────────── alignment to the published annotations ─────────────────────────

def _boxes(anns: list) -> list:
    return sorted((a["category_id"], *(round(float(v), 1) for v in a["bbox"])) for a in anns)


def _same(a: list, b: list, tol: float = 1.0) -> bool:
    return len(a) == len(b) and all(x[0] == y[0] and all(abs(p - q) <= tol for p, q in zip(x[1:], y[1:])) for x, y in zip(a, b))


def align(published: dict, pool: dict) -> tuple[dict, dict]:
    """published: COCO dict of one split; pool: {file_name: (image dir, our annotations, (h, w))} of every converted frame.
    Returns (COCO dict of the kept frames with the published annotations, report)."""
    by = {}
    for a in published["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    imgs, anns, dirs = [], [], {}
    rep = {"published": len(published["images"]), "kept": 0, "missing": [], "size_mismatch": [], "label_mismatch": 0}
    for im in published["images"]:
        f = im["file_name"]
        if f not in pool:
            rep["missing"].append(f)
            continue
        d, ours, hw = pool[f]
        if hw != (im["height"], im["width"]):
            rep["size_mismatch"].append(f)
            continue
        pub = by.get(im["id"], [])
        rep["label_mismatch"] += not _same(_boxes(pub), _boxes(ours))
        iid = len(imgs)
        imgs.append({"id": iid, "file_name": f, "height": im["height"], "width": im["width"]})
        anns += [{"id": len(anns) + k, "image_id": iid, "category_id": a["category_id"], "bbox": list(a["bbox"]), "ignore": a.get("ignore", 0),
                  "iscrowd": a.get("iscrowd", 0)} for k, a in enumerate(pub)]
        dirs[f] = d
    rep["kept"] = len(imgs)
    return {"images": imgs, "annotations": anns, "categories": CATS, "_dirs": dirs}, rep


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:   # another file system: copy
        shutil.copyfile(src, dst)


def write_set(data: Path, name: str, coco: dict, img_dirs) -> Path:
    """<data>/sets/<name>/{ann.json, thermal/} + cache in <data>/cache. img_dirs: one dir or {file_name: dir}."""
    sd = data / "sets" / name
    sub = sd / "thermal"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    for im in coco["images"]:
        d = img_dirs[im["file_name"]] if isinstance(img_dirs, dict) else img_dirs
        _link(Path(d) / im["file_name"], sub / im["file_name"])
    clean = {k: v for k, v in coco.items() if not k.startswith("_")}
    clean.setdefault("categories", CATS)
    (sd / "ann.json").write_text(json.dumps(clean))
    cache = data / "cache"
    build_cache_thermal(sd, name, cache)
    shutil.copyfile(sd / "ann.json", cache / f"{name}_ann.json")
    return sd


def prepare(dataset: str, src: str | Path, out: str | Path, do_align: bool = True, annotations: str | Path = PUBLISHED) -> dict:
    """Convert one dataset and write its splits + caches. Returns a report per set."""
    src, data = Path(src), Path(out)
    if not src.exists():
        raise FileNotFoundError(f"--src {src} does not exist (expected: {SOURCES[dataset]['src']})")
    cands = convert(dataset, src, data / "work")
    pool = {}
    for coco, d in cands.values():
        by = {}
        for a in coco["annotations"]:
            by.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            pool[im["file_name"]] = (d, by.get(im["id"], []), (im["height"], im["width"]))
    report = {}
    for name, (coco, d) in cands.items():
        pub = Path(annotations) / f"{name}.json"
        if do_align and pub.exists():
            aligned, rep = align(json.loads(pub.read_text()), pool)
            if not aligned["images"]:
                raise RuntimeError(f"{name}: none of the {rep['published']} published images were produced; check --src")
            write_set(data, name, aligned, aligned["_dirs"])
            rep["aligned"] = True
        else:
            write_set(data, name, coco, d)
            rep = {"aligned": False, "kept": len(coco["images"]), **({"note": "no published annotation file"} if do_align else {})}
        rep["boxes"] = sum(1 for a in json.loads((data / "sets" / name / "ann.json").read_text())["annotations"] if a["category_id"] == 0)
        report[name] = rep
    return report


def _print_report(report: dict) -> None:
    for name, r in report.items():
        line = f"{name:20} frames {r['kept']:6}  persons {r['boxes']:6}"
        if r.get("aligned"):
            line += f"  (published {r['published']}, missing {len(r['missing'])}, size mismatch {len(r['size_mismatch'])}"
            line += f", label mismatch {r['label_mismatch']})"
        elif r.get("note"):
            line += f"  ({r['note']}: converter split)"
        print(line, flush=True)
        for f in r.get("missing", [])[:10]:
            print(f"    missing: {f}")
        if len(r.get("missing", [])) > 10:
            print(f"    ... and {len(r['missing']) - 10} more missing")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=sources_text(), formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sources", help="print download URLs and licences of the datasets")
    for k, s in SOURCES.items():
        p = sub.add_parser(k, help=f"{s['title']} ({s['license']}, {s['url']})", description=f"{s['title']}\nlicence {s['license']} · {s['url']}",
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("--src", required=True, help=f"original download: {s['src']}")
        p.add_argument("--out", required=True, help="data root (sets/, cache/, work/ are created inside)")
        p.add_argument("--no-align", action="store_true", help="keep the converter's own split instead of the published annotation files")
        p.add_argument("--annotations", default=str(PUBLISHED), help="folder with the published <set>.json files (default: data/annotations of this repo)")
    a = ap.parse_args(argv)
    if a.cmd == "sources":
        print(sources_text())
        return 0
    rep = prepare(a.cmd, a.src, a.out, not a.no_align, a.annotations)
    _print_report(rep)
    (Path(a.out) / "sets").mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "sets" / f"prepare_{a.cmd}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
