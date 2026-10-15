"""ONNX evaluation and release export of hae models.

- `eval_onnx` / `detect_onnx`: run an exported ONNX model over cached frames (AP metrics or raw detections).
- `prf_table`: recall / precision / F1 / false positives per frame at a grid of confidence thresholds (IoU >= 0.5).
- `release_model`: checkpoint -> fp32 ONNX + int8 ONNX (calibrated on real training tiles, cls/box head kept fp32)
  with embedded metadata (recommended scale and confidence thresholds).
Everything is built from our own config and weights; no third-party weights are involved.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from hae.data.cache import Cache
from hae.data.sampler import load_frames
from hae.decode_np import decode_np
from hae.metrics import evaluate_detections, evaluate_protocol
from hae.recall import recall_at

THRS = tuple(round(0.05 * i, 2) for i in range(1, 13))   # 0.05 ... 0.60
INTENDED_USE = "Civilian search and rescue: decision support for detecting people in aerial imagery. Not a sole basis for life-critical decisions."


def _session(path, threads: int = 1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _scaled(rgb, t, scale: float):
    if scale == 1.0:
        return rgb, t
    return tuple(torch.nn.functional.interpolate(x, scale_factor=scale, mode="bilinear", align_corners=False) for x in (rgb, t))


def detect_onnx(onnx_path, cache: Cache, frames, mode: str, tnorm: bool, stride: int = 4, scale: float = 1.0, score_thr: float = 0.01, pre_lnorm: int = 0) -> list:
    """Detections of an ONNX model per frame, in frame coordinates (for recall / false-alarm tables)."""
    s = _session(onnx_path, threads=2)
    names = {i.name for i in s.get_inputs()}
    out = []
    for i in frames:
        rgb, t = _scaled(*load_frames(cache, [i], t_norm=tnorm, pre_lnorm=pre_lnorm), scale)
        cls, box = s.run(None, {k: v.numpy() for k, v in (("rgb", rgb), ("t", t)) if k in names})
        for d in decode_np(cls, box, stride, topk=300, score_thr=score_thr):
            d = d.copy()
            d[:, :4] /= scale   # upscaled input: boxes back to original coordinates
            out.append(d)
    return out


def eval_onnx(onnx_path, cache: Cache, anns, frames, mode: str, tnorm: bool, ncls: int, stride: int = 4, scale: float = 1.0, pre_lnorm: int = 0) -> dict:
    """COCO-style AP metrics of an ONNX model on cached frames (protocol metrics per class when ncls > 1)."""
    frames = list(frames)
    dets = detect_onnx(onnx_path, cache, frames, mode, tnorm, stride, scale, 0.01, pre_lnorm)
    gts, igs, labs = [anns[i].boxes for i in frames], [anns[i].ignore for i in frames], [anns[i].labels for i in frames]
    if ncls > 1:
        return evaluate_protocol(gts, labs, igs, dets, (cache.h, cache.w))
    return evaluate_detections(gts, igs, dets, (cache.h, cache.w))


def prf_table(gts, igs, dets, thrs=THRS) -> dict:
    """Per confidence threshold: recall, precision, F1, FP per frame (IoU >= 0.5, class-agnostic; ignore regions are not penalised)."""
    out = {}
    for t in thrs:
        m = recall_at(gts, igs, dets, t)
        m["f1"] = 2 * m["recall"] * m["precision"] / max(m["recall"] + m["precision"], 1e-9)
        out[str(t)] = m
    return out


def _latency_ms(path, hw, n: int = 25) -> float:
    """Median latency of one tile, 1 thread (the deployment setting)."""
    s = _session(path, 1)
    feeds = {i.name: np.random.rand(1, 3 if i.name == "rgb" else 1, *hw).astype(np.float32) for i in s.get_inputs()}
    for _ in range(3):
        s.run(None, feeds)
    t = []
    for _ in range(n):
        a = time.perf_counter()
        s.run(None, feeds)
        t.append(time.perf_counter() - a)
    return round(1000 * float(np.median(t)), 2)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calibration_tiles(caches, mode: str, n: int, scale: float = 1.0, tnorm: bool = False, pre_lnorm: int = 0, hw=(256, 320), seed: int = 0) -> list:
    """Random tiles from the given (training) caches in turn, at the deployment scale: int8 calibration must see the
    same data distribution and object scale as inference."""
    caches = list(caches)
    if not caches:
        raise ValueError("at least one calibration cache is required")
    rng, out, th, tw = np.random.default_rng(seed), [], *hw
    for k in range(n):
        c = caches[k % len(caches)]
        rgb, t = _scaled(*load_frames(c, [int(rng.integers(c.n))], t_norm=tnorm, pre_lnorm=pre_lnorm), scale)
        x = rgb if mode == "rgb" else t
        y, xx = int(rng.integers(0, max(x.shape[2] - th, 0) + 1)), int(rng.integers(0, max(x.shape[3] - tw, 0) + 1))
        out.append(np.ascontiguousarray(x[:, :, y:y + th, xx:xx + tw].numpy().astype(np.float32)))
    return out


def model_mode(cfg: dict) -> str:
    """Single input mode of a model from its training config: rgb or lwir."""
    modes = [m for m in cfg.get("eval_modes") or () if m in ("rgb", "lwir")]
    if modes:
        return modes[0]
    p = list(cfg.get("mode_probs") or (0, 0, 1))
    return "rgb" if p[1] > p[2] else "lwir"


def release_model(checkpoint, out_dir, name: str | None = None, mode: str | None = None, calib_caches=(), calib_n: int = 256,
                  scale: float | None = None, conf: float | None = None, sensitive_conf: float | None = None, tile_hw=(256, 320),
                  meta: dict | None = None, weights: bool = True) -> dict:
    """Export a checkpoint (`.pt` or released `.safetensors`) as `<name>_<mode>.onnx` and, given calibration caches,
    `<name>_<mode>_int8.onnx`; embeds metadata and (for `.pt`, weights=True) writes `<name>.safetensors` + `config.json`.
    Returns a report (also written to out_dir/report.json)."""
    from torch.utils.flop_counter import FlopCounterMode

    from hae.checkpoint import load_checkpoint, save_safetensors
    from hae.export import export_onnx
    from hae.infer import default_metadata, embed_metadata
    from hae.quantize import quantize_int8

    checkpoint, out = Path(checkpoint), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, cfg = load_checkpoint(checkpoint)
    name = name or checkpoint.stem
    mode = mode or model_mode(cfg)
    if mode not in ("rgb", "lwir"):
        raise ValueError(f"mode must be rgb or lwir, got {mode!r}")
    scale = float(cfg.get("eval_scale", 1.0) if scale is None else scale)
    tnorm, pre = bool(getattr(model.cfg, "tnorm", False)), int(getattr(model.cfg, "pre_lnorm", 0) or 0)
    p32, p8 = out / f"{name}_{mode}.onnx", out / f"{name}_{mode}_int8.onnx"
    export_onnx(model, p32, mode, tile_hw)
    key = "rgb" if mode == "rgb" else "t"
    x = torch.rand(1, 3 if mode == "rgb" else 1, *tile_hw)
    ref = model.for_mode(mode).eval() if hasattr(model, "for_mode") else model
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        o = ref(**{key: x})
    cls, box = _session(p32).run(None, {key: x.numpy()})
    diff = float(max(np.abs(cls - o["cls"].numpy()).max(), np.abs(box - o["box"].numpy()).max()))
    files = [p32]
    if calib_caches:
        quantize_int8(p32, p8, {key: calibration_tiles(calib_caches, mode, calib_n, scale, tnorm, pre, tile_hw)})
        files.append(p8)
    base = {**default_metadata(mode, model.cfg.ncls, tnorm, model.cfg.stride, pre), "name": name, "recommended_scale": scale,
            "intended_use": INTENDED_USE, **({"recommended_conf": conf} if conf is not None else {}),
            **({"sensitive_conf": sensitive_conf} if sensitive_conf is not None else {}), **(meta or {})}
    for p in files:
        embed_metadata(p, {**base, "precision": "int8 (cls/box head kept fp32)" if p == p8 else "fp32"})
    if weights and checkpoint.suffix == ".pt":
        files.append(save_safetensors(model, cfg, out, name, {"mode": mode}))
    rep = {"name": name, "mode": mode, "checkpoint": str(checkpoint), "params": sum(p.numel() for p in model.parameters()),
           f"gmac_per_tile_{tile_hw[0]}x{tile_hw[1]}_at_x1": round(fc.get_total_flops() / 2e9, 4), "recommended_scale": scale,
           "recommended_conf": conf, "sensitive_conf": sensitive_conf, "torch_vs_onnx_fp32_max_abs_diff": diff,
           "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
           "ms_p50_1thread": {p.name: _latency_ms(p, tile_hw) for p in files if p.suffix == ".onnx"}}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep
