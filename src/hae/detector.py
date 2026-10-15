"""Simple user API: `Detector(model).predict(image)` -> Results; `Detector.detect(array)` -> array (n,6) [x1,y1,x2,y2,score,class].

Depends only on numpy, opencv, onnx and onnxruntime (no PyTorch). Input mode (rgb/lwir), preprocessing, stride and the recommended
input scale are read from the ONNX metadata. SAHI tiling and input upscaling are enabled with the `tile` and `scale` arguments.

Model lookup by name ("hae-s-ir"), first match wins: explicit `root` argument (only that directory) -> $HAE_MODELS -> ./models
(current directory) -> `models/` of a source checkout next to the package -> Hugging Face Hub download (needs huggingface_hub).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from hae import hub
from hae.decode_np import decode_np
from hae.infer import local_contrast_u8, preprocess_rgb, preprocess_thermal, read_metadata
from hae.sahi import sliced_predict


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
REPO_MODELS = Path(__file__).resolve().parents[2] / "models"     # src layout: <repo>/src/hae/detector.py -> <repo>/models


def search_roots(root: str | Path | None = None) -> list:
    """Local model directories in lookup order (existing ones only, except an explicit `root`, which is used as is)."""
    if root is not None:
        return [Path(root)]
    cand = ([Path(os.environ["HAE_MODELS"])] if os.environ.get("HAE_MODELS") else []) + [Path.cwd() / "models", REPO_MODELS]
    out = []
    for c in cand:
        if c.is_dir() and c.resolve() not in [o.resolve() for o in out]:
            out.append(c)
    return out


def models_root(root: str | Path | None = None) -> Path:
    """The first local model directory (explicit root, $HAE_MODELS, ./models, repository models/); ./models if none exists."""
    r = search_roots(root)
    return r[0] if r else Path("models")


def list_models(root: str | Path | None = None) -> list:
    r = models_root(root)
    return sorted(d.name for d in r.iterdir() if d.is_dir() and list(d.glob("*.onnx"))) if r.exists() else []


def resolve_model(model: str | Path, root: str | Path | None = None, int8: bool = True) -> Path:
    """Path to an .onnx file OR a model name ("hae-s-ir") looked up locally (see module docstring), then on the Hugging Face Hub."""
    p = Path(model)
    if p.suffix == ".onnx":
        if p.exists():
            return p
        raise FileNotFoundError(f"model file not found: {p}")
    name = str(model)
    rel = hub.model_filename(name, int8)        # release suffixes (hae-n-ir-ext-kdf) do not change the mode
    roots = search_roots(root)
    for r in roots:
        if (r / rel).exists():
            return r / rel
    where = ", ".join(str(r) for r in roots) or "(no local model directory)"
    avail = ", ".join(list_models(root)) or "(none)"
    if root is None and name in hub.RELEASED_MODELS:
        if hub.available():
            return hub.download(name, int8)
        raise FileNotFoundError(f"model not found: {model}. Searched: {where}. Options: pass an .onnx path, set root= or $HAE_MODELS, "
                                f"put models in ./models, or install huggingface_hub (pip install \"hae[hub]\") to download it automatically.")
    raise FileNotFoundError(f"model not found: {model}. Available in {where}: {avail}. Released models: {', '.join(hub.RELEASED_MODELS)}")


class Detector:
    """`Detector("hae-s-ir")` or `Detector("path/model.onnx")`; `predict(source)` -> Results (one image) or a list (folder/list)."""

    def __init__(self, model: str | Path, conf: float | None = None, scale: float | None = None, tile: tuple | None = None,
                 overlap: float = 0.2, threads: int = 1, iou_thr: float = 0.5, root: str | Path | None = None, preset: str = "balanced"):
        self.path = resolve_model(model, root)
        self.meta = read_metadata(self.path)
        self.mode = self.meta.get("mode", "lwir")
        self.tnorm = bool(self.meta.get("tnorm", False))
        self.pre_lnorm = int(self.meta.get("pre_lnorm", 0) or 0)   # model trained on local_contrast_u8(k) frames; absent key = plain frames
        self.stride = int(self.meta.get("stride", 4))
        self.names = self.meta.get("class_names", ["person"])
        if preset not in ("balanced", "sensitive"):
            raise ValueError(f"preset must be balanced or sensitive, got {preset!r}")
        key = "recommended_conf" if preset == "balanced" else "sensitive_conf"   # balanced = max F1, sensitive = max F2 (recall weighs more)
        self.conf = float(conf if conf is not None else self.meta.get(key, self.meta.get("recommended_conf", 0.3)))   # the default threshold is per model (confidence calibration differs)
        self.tile, self.overlap, self.iou_thr = tile, float(overlap), float(iou_thr)
        self.interp = cv2.INTER_LINEAR   # interpolation for input upscaling (cubic/Lanczos gave no gain)
        self.scale = float(scale if scale is not None else self.meta.get("recommended_scale", 1.0))
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    @classmethod
    def from_pretrained(cls, name: str, root: str | Path | None = None, int8: bool = True, **kw) -> "Detector":
        return cls(resolve_model(name, root, int8), **kw)

    def _run(self, x: np.ndarray) -> np.ndarray:
        """x: (1,C,h,w) -> detections (n,6) in x coordinates (pixels, class)."""
        h, w = x.shape[-2:]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)))
        cls, box = self.sess.run(None, {self.input: np.ascontiguousarray(x, dtype=np.float32)})
        d = decode_np(cls, box, self.stride, topk=300, score_thr=self.conf)[0]
        d = np.asarray(d, np.float32).reshape(len(d), -1) if len(d) else np.zeros((0, 6), np.float32)
        if d.shape[1] == 5:
            d = np.concatenate([d, np.zeros((len(d), 1), np.float32)], 1)
        d[:, [0, 2]] = d[:, [0, 2]].clip(0, w)
        d[:, [1, 3]] = d[:, [1, 3]].clip(0, h)
        return d

    def detect(self, image: np.ndarray, scale: float | None = None, flip: bool = False) -> np.ndarray:
        """Low level: image array -> (n,6) [x1,y1,x2,y2,score,class] (the same pipeline as predict).
        `scale` overrides the input scale for this call only; `flip` mirrors horizontally (for TTA), boxes are mapped back to the original."""
        sc = self.scale if scale is None else float(scale)
        img = np.ascontiguousarray(np.asarray(image)[:, ::-1]) if flip else np.asarray(image)
        h0, w0 = img.shape[:2]
        t_pre = time.perf_counter()
        if self.mode == "lwir" and self.pre_lnorm > 0:
            img = local_contrast_u8(img, self.pre_lnorm)                                            # on the FULL frame, before scaling and tiling
        x = preprocess_thermal(img, self.tnorm) if self.mode == "lwir" else preprocess_rgb(img)     # per-frame normalisation BEFORE scaling and tiling
        if sc != 1.0:
            hw = (int(round(w0 * sc)), int(round(h0 * sc)))
            x = np.stack([cv2.resize(c, hw, interpolation=self.interp) for c in x[0]])[None]
        t_inf = time.perf_counter()
        if self.tile:
            fr = x[0]
            infer = (lambda r, t, o: self._run(r[None])) if self.mode == "rgb" else (lambda r, t, o: self._run(t[None]))
            d = sliced_predict(infer, fr if self.mode == "rgb" else None, fr if self.mode == "lwir" else None,
                               tile_hw=tuple(self.tile), overlap=self.overlap, iou_thr=self.iou_thr)
            d = d.reshape(len(d), -1)
            if d.shape[1] == 5:
                d = np.concatenate([d, np.zeros((len(d), 1), np.float32)], 1)
        else:
            d = self._run(x)
        t_post = time.perf_counter()
        d = d.copy()
        d[:, :4] /= sc
        d[:, [0, 2]] = d[:, [0, 2]].clip(0, w0)
        d[:, [1, 3]] = d[:, [1, 3]].clip(0, h0)
        if flip:
            d[:, [0, 2]] = w0 - d[:, [2, 0]]
        out = d[np.argsort(-d[:, 4])] if len(d) else d
        self.last_speed = {"preprocess": 1000 * (t_inf - t_pre), "inference": 1000 * (t_post - t_inf), "postprocess": 1000 * (time.perf_counter() - t_post)}
        return out

    def _load(self, path: str | Path) -> np.ndarray:
        from PIL import Image
        im = Image.open(path)
        return np.asarray(im.convert("RGB") if self.mode == "rgb" else im.convert("L"))

    def predict(self, source):
        """source: image array | file path | folder | list (of paths or arrays). One image -> Results, otherwise a list of Results."""
        from hae.results import Results
        one = isinstance(source, np.ndarray) or (isinstance(source, (str, Path)) and Path(source).is_file())
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            items = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        elif one:
            items = [source]
        else:
            items = list(source)
        out = []
        for it in items:
            img, path = (it, None) if isinstance(it, np.ndarray) else (self._load(it), str(it))
            r = Results.from_array(img, self.detect(img), self.names, path, self.mode)
            r.speed = dict(self.last_speed)
            out.append(r)
        return out[0] if one else out
