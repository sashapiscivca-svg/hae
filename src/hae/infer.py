"""Preprocessing and ONNX metadata for inference WITHOUT PyTorch (numpy + onnx/onnxruntime).

Preprocessing matches training exactly: thermal = per-frame z-score (t - mean) / max(std, 1) with statistics on the 0..255 scale;
RGB = pixel / 255. Mode parameters are stored in the ONNX metadata, so each model file is self-contained.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VERSION = "1.0.0"
CLASS_NAMES = {1: ["person"], 3: ["person", "rider", "crowd"]}


def preprocess_thermal(t: np.ndarray, tnorm: bool = True) -> np.ndarray:
    """t: (H,W), (H,W,1) or (H,W,3) uint8 -> (1,1,H,W) float32."""
    a = np.asarray(t)
    if a.ndim == 3:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"a thermal frame must be (H,W), (H,W,1) or (H,W,3), got {a.shape}")
    f = a.astype(np.float32)
    f = (f - f.mean()) / max(float(f.std()), 1.0) if tnorm else f / 255.0
    return f[None, None]


def local_contrast_u8(g: np.ndarray, k: int = 61) -> np.ndarray:
    """Local contrast normalisation of a thermal frame: uint8 (H,W) -> uint8 (H,W).
    out = clip(128 + 32 * (f - m) / (sd + 4)), where m, sd are the mean and std in a k x k window (cv2.blur, BORDER_REFLECT_101).
    Removes slow background (warm ground, vignetting) and equalises scene contrast; applied to the FULL frame before any crop,
    scaling or tiling, identically in training, evaluation and ONNX inference (`pre_lnorm` in the metadata)."""
    import cv2
    a = np.asarray(g)
    if a.ndim == 3:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"a thermal frame must be (H,W), (H,W,1) or (H,W,3), got {a.shape}")
    k = int(k)
    f = a.astype(np.float32)
    m = cv2.blur(f, (k, k))
    sd = np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - m * m, 1))
    return np.clip(128 + 32 * (f - m) / (sd + 4), 0, 255).astype(np.uint8)


def preprocess_rgb(img: np.ndarray) -> np.ndarray:
    """img: (H,W,3) RGB uint8 -> (1,3,H,W) float32 in 0..1."""
    a = np.asarray(img)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"an RGB frame must be (H,W,3), got {a.shape}")
    return (a.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def read_metadata(path: str | Path) -> dict:
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    out = {}
    for p in m.metadata_props:
        try:
            out[p.key] = json.loads(p.value)
        except ValueError:
            out[p.key] = p.value
    return out


def embed_metadata(path: str | Path, meta: dict) -> None:
    """Add/update ONNX metadata (values are JSON-encoded); other existing keys are kept."""
    import onnx
    m = onnx.load(str(path))
    cur = {p.key: p.value for p in m.metadata_props}
    cur.update({k: json.dumps(v, ensure_ascii=False) for k, v in meta.items()})
    del m.metadata_props[:]
    for k, v in cur.items():
        e = m.metadata_props.add()
        e.key, e.value = k, v
    onnx.save(m, str(path))


_THERMAL_Z = "per-frame z-score (t - mean) / max(std, 1), statistics on the 0..255 scale"
_RGB = "pixel / 255, channel order RGB"


def default_metadata(mode: str, ncls: int, tnorm: bool, stride: int, pre_lnorm: int = 0) -> dict:
    """pre_lnorm > 0: the model was trained on frames after local_contrast_u8(k=pre_lnorm); the key is written only then,
    so metadata of other models is unchanged."""
    t_rule = _THERMAL_Z if tnorm else "t / 255"
    if int(pre_lnorm) > 0:
        t_rule = f"local_contrast_u8(k={int(pre_lnorm)}) on the full uint8 frame first, then {t_rule}"
    pre = {"lwir": f"thermal (LWIR, 8-bit grayscale): {t_rule}; input shape (B,1,H,W)",
           "rgb": f"RGB: {_RGB}, input shape (B,3,H,W)"}.get(mode, f"two inputs: rgb = {_RGB}, shape (B,3,H,W); t = thermal (LWIR, 8-bit grayscale), {t_rule}, shape (B,1,H,W)")
    meta = {"hae_version": VERSION, "mode": mode, "tnorm": bool(tnorm), "stride": int(stride), "ncls": int(ncls),
            "class_names": CLASS_NAMES.get(ncls, [f"c{i}" for i in range(ncls)]), "preprocess": pre, "input_multiple_of": 32,
            "outputs": "cls: class logits (B,ncls,H/stride,W/stride); box: raw distances l,t,r,b (B,4,H/stride,W/stride). "
                       "Decode: 3x3 peak picking on sigmoid(cls), box = softplus(raw) * stride from the cell centre (see hae.decode_np)",
            "recommended_scale": 1.0, "license": "MIT"}
    if mode != "rgb" and int(pre_lnorm) > 0:
        meta["pre_lnorm"] = int(pre_lnorm)
    return meta
