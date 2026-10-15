"""uint8 memmap frame cache: training reads crops without JPEG decoding."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from hae.data.annotations import load_annotations


def _fingerprint(ann: Path) -> str:
    """Cache provenance: absolute path and size of the annotation file. A cache is reused only for the same source."""
    return f"{ann.resolve()}:{ann.stat().st_size}"


def _reusable(meta_path: Path, fp: str) -> bool:
    if not meta_path.exists():
        return False
    try:
        return json.loads(meta_path.read_text()).get("source") == fp
    except (OSError, ValueError):
        return False


def _build_rgb(ann_file: Path, rgb_dir: Path, t_dir: Path | None, nm: str, out: Path, reuse: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(ann_file)
    if reuse and _reusable(out / f"{nm}_meta.json", fp):   # cache of this source already built (meta is written AFTER filling): skip
        return
    rgb_only = False
    anns = load_annotations(ann_file)
    # nominal size = the most frequent one in the JSON (a few frames may carry stale size metadata)
    (h, w), _ = Counter((a.height, a.width) for a in anns).most_common(1)[0]
    n = len(anns)
    rgb = np.memmap(out / f"{nm}_rgb.u8", np.uint8, "w+", shape=(n, h, w, 3))
    t = np.memmap(out / f"{nm}_t.u8", np.uint8, "w+", shape=(n, h, w))
    for i, a in enumerate(anns):
        r = np.asarray(Image.open(rgb_dir / a.file_name).convert("RGB"))
        tp = t_dir / a.file_name if t_dir is not None else None
        if tp is not None and tp.exists():
            th = np.asarray(Image.open(tp).convert("L"))
        else:   # RGB-only set: the "thermal" channel = RGB luminance, training modes restricted to rgb
            th, rgb_only = np.asarray(Image.fromarray(r).convert("L")), True
        if r.shape[:2] != (h, w) or th.shape != (h, w):
            raise ValueError(f"{a.file_name}: RGB size {r.shape[:2]} / LWIR {th.shape} != expected {(h, w)}")
        rgb[i], t[i] = r, th
    rgb.flush()
    t.flush()
    meta = {"n": n, "h": h, "w": w, "source": fp, **({"modes": ["rgb"]} if rgb_only else {})}
    (out / f"{nm}_meta.json").write_text(json.dumps(meta))


def build_cache(root: str | Path, split: str, out_dir: str | Path, name: str | None = None, reuse: bool = True) -> None:
    """Cache of an RGBTDronePerson-style root: <root>/<split>_thermal.json + <root>/<split>/{visible,thermal}/.
    name = prefix of the cache files (default: the split name), so several sets can live in one cache directory."""
    root = Path(root)
    _build_rgb(root / f"{split}_thermal.json", root / split / "visible", root / split / "thermal", name or split, Path(out_dir), reuse)


def build_cache_rgb(set_dir: str | Path, name: str, out_dir: str | Path, reuse: bool = True) -> None:
    """Cache of an RGB-only set in the layout of scripts/prepare_data.py: <set_dir>/ann.json + <set_dir>/visible/."""
    d = Path(set_dir)
    _build_rgb(d / "ann.json", d / "visible", None, name, Path(out_dir), reuse)


def build_cache_thermal(unified_dir: str | Path, name: str, out_dir: str | Path, reuse: bool = True) -> None:
    """Cache of a thermal-only set (<unified_dir>/ann.json + thermal/): <name>_t.u8 + <name>_meta.json, no RGB."""
    root, out = Path(unified_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(root / "ann.json")
    if reuse and _reusable(out / f"{name}_meta.json", fp):
        return
    anns = load_annotations(root / "ann.json")
    (h, w), _ = Counter((a.height, a.width) for a in anns).most_common(1)[0]
    t = np.memmap(out / f"{name}_t.u8", np.uint8, "w+", shape=(len(anns), h, w))
    for i, a in enumerate(anns):
        th = np.asarray(Image.open(root / "thermal" / a.file_name).convert("L"))
        if th.shape != (h, w):
            raise ValueError(f"{a.file_name}: size {th.shape} != expected {(h, w)}")
        t[i] = th
    t.flush()
    (out / f"{name}_meta.json").write_text(json.dumps({"n": len(anns), "h": h, "w": w, "has_rgb": False, "source": fp}))


class Cache:
    def __init__(self, out_dir: str | Path, split: str):
        out = Path(out_dir)
        m = json.loads((out / f"{split}_meta.json").read_text())
        self.n, self.h, self.w = m["n"], m["h"], m["w"]
        self.modes = tuple(m.get("modes", ("both", "rgb", "lwir")))   # training modes allowed on this cache
        self._stats_path, self._tstats = out / f"{split}_tstats.npy", None
        hp = out / f"{split}_halo.u8"
        self.halo = np.memmap(hp, np.uint8, "r", shape=(self.n, self.h, self.w)) if hp.exists() else None   # HALO masks
        has_rgb = (out / f"{split}_rgb.u8").exists()     # thermal-only sets have no RGB
        self.rgb = np.memmap(out / f"{split}_rgb.u8", np.uint8, "r", shape=(self.n, self.h, self.w, 3)) if has_rgb else None
        self.t = np.memmap(out / f"{split}_t.u8", np.uint8, "r", shape=(self.n, self.h, self.w))
        self.pre_lnorm = lnorm_k_of(out)   # >0: thermal frames in this cache already went through local_contrast_u8(k)


LNORM_MARK = "lnorm.json"


def lnorm_k_of(cache_dir: str | Path) -> int:
    """k of the local normalisation applied to the cached thermal frames (lnorm.json marker), or 0 for a raw cache."""
    p = Path(cache_dir) / LNORM_MARK
    try:
        return int(json.loads(p.read_text())["k"]) if p.exists() else 0
    except (OSError, ValueError, KeyError):
        return 0


def lnorm_cache(src_dir: str | Path, dst_dir: str | Path, k: int = 61, verbose: bool = False) -> Path:
    """Copy of a cache with thermal frames after local_contrast_u8(k) on FULL frames.
    *_t.u8 are transformed frame by frame (memory: one frame); RGB memmaps (*_rgb.u8) are symlinked (unchanged);
    *_tstats.npy and *_halo.u8 are NOT copied: frame statistics and HALO masks are recomputed from the transformed frames;
    other files (metadata, annotations) are copied. The lnorm.json marker is written last and lists processed sets:
    a repeated call processes only new *_t.u8 (e.g. held-out sets added later)."""
    import shutil
    from hae.infer import local_contrast_u8
    src, dst, k = Path(src_dir), Path(dst_dir), int(k)
    if src.resolve() == dst.resolve():
        raise ValueError("lnorm_cache: source and destination directories must differ")
    if lnorm_k_of(src):
        raise ValueError(f"{src} is already transformed (k={lnorm_k_of(src)}); double normalisation is not allowed")
    dst.mkdir(parents=True, exist_ok=True)
    mark = dst / LNORM_MARK
    done = set(json.loads(mark.read_text()).get("sets", [])) if lnorm_k_of(dst) == k else set()
    if lnorm_k_of(dst) not in (0, k):
        raise ValueError(f"{dst} was transformed with a different k={lnorm_k_of(dst)}")
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.name == LNORM_MARK or f.name.endswith(("_tstats.npy", "_halo.u8")):
            continue
        d = dst / f.name
        if f.name.endswith("_t.u8"):
            nm = f.name[:-len("_t.u8")]
            if nm in done and d.exists() and d.stat().st_size == f.stat().st_size:
                continue
            m = json.loads((src / f"{nm}_meta.json").read_text())
            shp = (m["n"], m["h"], m["w"])
            a = np.memmap(f, np.uint8, "r", shape=shp)
            b = np.memmap(d, np.uint8, "w+", shape=shp)
            for i in range(shp[0]):
                b[i] = local_contrast_u8(a[i], k)
            b.flush()
            del a, b
            for st in (dst / f"{nm}_tstats.npy", dst / f"{nm}_halo.u8"):   # stale after the transform
                st.unlink(missing_ok=True)
            done.add(nm)
            if verbose:
                print(f"lnorm k={k}: {nm} ({shp[0]} frames)", flush=True)
        elif f.name.endswith("_rgb.u8"):
            if not d.exists():
                d.symlink_to(f.resolve())
        else:
            shutil.copy2(f, d)
    mark.write_text(json.dumps({"k": k, "source": str(src.resolve()), "sets": sorted(done)}))
    return dst


def ensure_lnorm_cache(cache_dir: str | Path, k: int, verbose: bool = False) -> Path:
    """Cache directory for a model with pre_lnorm=k: cache_dir itself if it is already transformed with the same k,
    otherwise the sibling <cache_dir>_lnorm<k> (built/extended by lnorm_cache)."""
    c = Path(cache_dir)
    kk = lnorm_k_of(c)
    if kk == int(k):
        return c
    if kk:
        raise ValueError(f"cache {c} was transformed with k={kk}, but k={k} is required")
    return lnorm_cache(c, c.parent / f"{c.name}_lnorm{int(k)}", k, verbose)


def _tstats(self):
    """(mean[N], std[N]) of thermal brightness per frame; computed once and cached in <split>_tstats.npy.
    Identical for training, evaluation and SAHI tiles: the normalisation depends on the frame, not the crop."""
    if self._tstats is None:
        if self._stats_path.exists():
            self._tstats = tuple(np.load(self._stats_path))
        else:
            mean = np.empty(self.n, np.float32)
            std = np.empty(self.n, np.float32)
            for i in range(self.n):
                a = np.asarray(self.t[i], np.float32)
                mean[i], std[i] = a.mean(), a.std()
            self._tstats = (mean, std)
            try:
                np.save(self._stats_path, np.stack([mean, std]))
            except OSError:
                pass
    return self._tstats


Cache.tstats = property(_tstats)


def build_halo_cache(cache: "Cache", anns, name: str, out_dir: str | Path) -> None:
    """HALO masks (hae.data.halo) for all positives of each frame -> <name>_halo.u8 (uint8, n×h×w)."""
    from hae.data.halo import halo_mask
    out = Path(out_dir)
    m = np.memmap(out / f"{name}_halo.u8", np.uint8, "w+", shape=(cache.n, cache.h, cache.w))
    for i in range(cache.n):
        m[i] = np.round(255 * halo_mask(np.asarray(cache.t[i]), anns[i].boxes)).astype(np.uint8)
    m.flush()
